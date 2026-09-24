from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import tempfile

import numpy as np


@dataclass
class Transition:
    """Single env step stored in the replay buffer."""

    image: np.ndarray  # uint8 (H, W, C)
    state: np.ndarray  # float32 vector
    action: np.ndarray  # float32
    reward: float
    done: bool
    next_image: np.ndarray  # kept in API; not duplicated in RAM storage
    next_state: np.ndarray
    terminated: bool | None = None  # None supports legacy callers: use done.


class ReplayBuffer:
    """
    Sequence-capable replay buffer for World Model training.

    Stores (image, state, action, reward, done) only. next_* is derived from
    the following index when sampling (saves ~2x image RAM).
    """

    def __init__(
        self,
        capacity: int,
        image_shape: tuple[int, ...],
        state_dim: int,
        action_dim: int,
        sequence_length: int = 64,
    ) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.image_shape = tuple(image_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.checkpoint_step: int | None = None
        # Sparse terminal successors plus the latest transition's successor.
        self._term_image: dict[int, np.ndarray] = {}
        self._term_state: dict[int, np.ndarray] = {}

        self.idx = 0
        self.full = False
        self.episode_start = np.ones((capacity,), dtype=np.bool_)
        self._valid_starts_cache: np.ndarray | None = None
        self._valid_starts_key: tuple[int, int, bool, int] | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    # ------------------------------------------------------------------
    # Online Training: convert read-only memmap to writable arrays
    # ------------------------------------------------------------------
    def make_writable(self, new_capacity: int | None = None) -> None:
        """Copy read-only (memmap) arrays into writable numpy arrays.

        If *new_capacity* exceeds the current size, all backing arrays are
        expanded so that new transitions can be appended via :meth:`add`.
        """
        old_size = len(self)
        new_cap = max(int(new_capacity or self.capacity), old_size)
        # Unwrap oldest -> newest before expanding a full ring. Otherwise a
        # saved write pointer in the middle becomes a false temporal adjacency.
        order = np.arange(old_size)
        if self.full and new_cap != self.capacity:
            order = (order + self.idx) % old_size
        index_map = {int(old): new for new, old in enumerate(order)}

        def _expand(src: np.ndarray, shape: tuple, dtype) -> np.ndarray:
            dst = np.zeros(shape, dtype=dtype)
            dst[:old_size] = src[order]
            return dst

        self.images = _expand(
            self.images, (new_cap, *self.image_shape), np.uint8
        )
        self.states = _expand(self.states, (new_cap, self.state_dim), np.float32)
        self.actions = _expand(
            self.actions, (new_cap, self.action_dim), np.float32
        )
        self.rewards = _expand(self.rewards, (new_cap,), np.float32)
        self.dones = _expand(self.dones, (new_cap,), np.bool_)
        self.terminated = _expand(self.terminated, (new_cap,), np.bool_)

        new_ep = np.ones((new_cap,), dtype=np.bool_)
        new_ep[:old_size] = self.episode_start[order]
        self.episode_start = new_ep

        if new_cap != self.capacity:
            self._term_image = {index_map[k]: v for k, v in self._term_image.items() if k in index_map}
            self._term_state = {index_map[k]: v for k, v in self._term_state.items() if k in index_map}
            self.idx = old_size % new_cap
            self.full = old_size >= new_cap
        self.capacity = new_cap
        self._valid_starts_cache = None
        self._valid_starts_key = None
        self._sample_img_buf = None

        # Drop stale memmap reference
        if hasattr(self, "_mmap_path"):
            del self._mmap_path

    def add(self, transition: Transition, *, is_first: bool = False) -> None:
        if np.shape(transition.image) != self.image_shape:
            raise ValueError(f"Replay image shape {np.shape(transition.image)} != {self.image_shape}")
        if len(self):
            previous = (self.idx - 1) % self.capacity
            if is_first and not self.dones[previous]:
                # Reset after eval/resume is a boundary, not a true terminal.
                self.dones[previous] = True
            elif not self.dones[previous]:
                self._term_image.pop(previous, None)
                self._term_state.pop(previous, None)
        i = self.idx
        # Overwriting a slot: drop sparse terminal cache for that index
        self._term_image.pop(i, None)
        self._term_state.pop(i, None)

        self.images[i] = transition.image
        self.states[i] = transition.state
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.dones[i] = transition.done
        self.terminated[i] = transition.done if transition.terminated is None else transition.terminated
        # Keep terminal observations AND the latest successor (no following row yet).
        self._term_image[i] = np.asarray(transition.next_image, dtype=np.uint8).copy()
        self._term_state[i] = np.asarray(transition.next_state, dtype=np.float32).copy()
        self.episode_start[i] = is_first

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
        self._valid_starts_cache = None

    def _valid_start_indices(self) -> np.ndarray:
        """
        Starts for contiguous sequences of length L in the buffer.
        Sequences may cross episode boundaries because `is_first` resets
        the RSSM recurrent state.
        """
        n = len(self)
        L = self.sequence_length
        key = (n, L, self.full, self.idx)
        if self._valid_starts_cache is not None and self._valid_starts_key == key:
            return self._valid_starts_cache

        if n < L:
            starts = np.array([], dtype=np.int64)
        elif not self.full:
            starts = np.arange(n - L + 1, dtype=np.int64)
        else:
            # Full circular buffer: avoid crossing self.idx (write pointer)
            all_starts = []
            if self.idx >= L:
                all_starts.append(np.arange(0, self.idx - L + 1))
            if self.capacity - self.idx >= L:
                all_starts.append(np.arange(self.idx, self.capacity - L + 1))
            starts = (
                np.concatenate(all_starts).astype(np.int64)
                if all_starts
                else np.array([], dtype=np.int64)
            )

        self._valid_starts_cache = starts
        self._valid_starts_key = key
        return starts

    def _next_image_at(self, t: int) -> np.ndarray:
        if t in self._term_image:
            return self._term_image[t]
        return self.images[(t + 1) % self.capacity]

    def _next_state_at(self, t: int) -> np.ndarray:
        if t in self._term_state:
            return self._term_state[t]
        return self.states[(t + 1) % self.capacity]

    def sample(
        self,
        batch_size: int,
        *,
        include_next: bool = False,
        term_ratio: float = 0.25,
    ) -> dict[str, np.ndarray]:
        starts = self._valid_start_indices()
        if batch_size <= 0 or not 0 <= term_ratio <= 1:
            raise ValueError("batch_size must be positive and term_ratio must be in [0, 1]")
        if include_next:
            # Old archives may omit the newest successor / terminal observations.
            bad = np.zeros(len(self), dtype=np.int64)
            missing = [int(i) for i in np.flatnonzero(self.dones[:len(self)]) if i not in self._term_image]
            newest = (self.idx - 1) % self.capacity
            if newest not in self._term_image:
                missing.append(newest)
            bad[missing] = 1
            bad_prefix = np.concatenate([[0], np.cumsum(bad)])
            starts = starts[(bad_prefix[starts + self.sequence_length] - bad_prefix[starts]) == 0]
        if len(starts) == 0:
            raise RuntimeError(
                "Not enough contiguous sequences in buffer. Collect more data."
            )
        L = self.sequence_length
        
        # Identify terminal sequences (containing at least one done)
        n = len(self)
        dones_int = self.dones[:n].astype(np.int32)
        prefix = np.concatenate([[0], np.cumsum(dones_int)])
        has_done = (prefix[starts + L] - prefix[starts]) > 0
        term_starts = starts[has_done]
        
        n_term = int(batch_size * term_ratio) if len(term_starts) > 0 else 0
        n_norm = batch_size - n_term
        
        chosen_norm = np.random.choice(starts, size=n_norm, replace=len(starts) < n_norm)
        if n_term > 0:
            chosen_term = np.random.choice(term_starts, size=n_term, replace=len(term_starts) < n_term)
            chosen = np.concatenate([chosen_norm, chosen_term])
        else:
            chosen = chosen_norm

        def gather(arr: np.ndarray) -> np.ndarray:
            # Preallocate one block (avoids list+stack peak / fragmentation)
            out_arr = np.empty((batch_size, L, *arr.shape[1:]), dtype=arr.dtype)
            for i, s in enumerate(chosen):
                out_arr[i] = arr[s : s + L]
            return out_arr

        # Reuse image scratch buffer across samples to reduce RAM fragmentation
        img_shape = (batch_size, L, *self.images.shape[1:])
        if (
            getattr(self, "_sample_img_buf", None) is None
            or self._sample_img_buf.shape != img_shape
        ):
            self._sample_img_buf = np.empty(img_shape, dtype=self.images.dtype)
        for i, s in enumerate(chosen):
            self._sample_img_buf[i] = self.images[s : s + L]

        out = {
            "image": self._sample_img_buf,
            "state": gather(self.states),
            "action": gather(self.actions),
            "reward": gather(self.rewards),
            "done": gather(self.dones),
            "terminated": gather(self.terminated),
            "is_first": gather(self.episode_start),
        }
        # WM needs successors; Actor-only posterior extraction does not.
        if include_next:
            out["next_image"] = np.stack(
                [np.stack([self._next_image_at(s + t) for t in range(L)], axis=0) for s in chosen],
                axis=0,
            )
            out["next_state"] = np.stack(
                [np.stack([self._next_state_at(s + t) for t in range(L)], axis=0) for s in chosen],
                axis=0,
            )
        return out

    def save(self, path: str | Path, *, checkpoint_step: int | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self)
        term_idx = np.array(sorted(self._term_image.keys()), dtype=np.int64)
        if len(term_idx):
            term_images = np.stack([self._term_image[i] for i in term_idx], axis=0)
            term_states = np.stack([self._term_state[i] for i in term_idx], axis=0)
        else:
            term_images = np.zeros((0, *self.image_shape), dtype=np.uint8)
            term_states = np.zeros((0, self.state_dim), dtype=np.float32)

        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            images=self.images[:n] if not self.full else self.images,
            states=self.states[:n] if not self.full else self.states,
            actions=self.actions[:n] if not self.full else self.actions,
            rewards=self.rewards[:n] if not self.full else self.rewards,
            dones=self.dones[:n] if not self.full else self.dones,
            terminated=self.terminated[:n] if not self.full else self.terminated,
            episode_start=self.episode_start[:n] if not self.full else self.episode_start,
            term_idx=term_idx,
            term_images=term_images,
            term_states=term_states,
            idx=np.array([self.idx]),
            full=np.array([self.full]),
            meta=np.array(
                [self.capacity, self.sequence_length, self.state_dim, self.action_dim],
                dtype=np.int64,
            ),
            image_shape=np.array(self.image_shape, dtype=np.int64),
            size=np.array([n], dtype=np.int64),
            checkpoint_step=np.array([-1 if checkpoint_step is None else checkpoint_step], dtype=np.int64),
        )
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """
        Load a bootstrap or online buffer. Capacity = actual size; images live in a disk
        memmap so RAM is not filled with a second full uint8 copy.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            _cap, sequence_length, state_dim, action_dim = data["meta"].tolist()
            image_shape = tuple(int(x) for x in data["image_shape"].tolist())
            imgs = data["images"]
            n = int(data["size"][0]) if "size" in data else int(imgs.shape[0])
            n = min(n, int(imgs.shape[0]))

            buf = object.__new__(cls)
            buf.capacity = n
            buf.sequence_length = int(sequence_length)
            buf.image_shape = image_shape
            buf.state_dim = int(state_dim)
            buf.action_dim = int(action_dim)
            buf._term_image = {}
            buf._term_state = {}
            buf._valid_starts_cache = None
            buf._valid_starts_key = None
            buf._sample_img_buf = None

            buf.states = np.array(data["states"][:n], dtype=np.float32, copy=True)
            buf.actions = np.array(data["actions"][:n], dtype=np.float32, copy=True)
            buf.rewards = np.array(data["rewards"][:n], dtype=np.float32, copy=True)
            buf.dones = np.array(data["dones"][:n], dtype=np.bool_, copy=True)
            buf.terminated = np.array(data["terminated"][:n] if "terminated" in data else buf.dones, dtype=np.bool_, copy=True)
            step = int(data["checkpoint_step"][0]) if "checkpoint_step" in data else -1
            buf.checkpoint_step = None if step < 0 else step
            buf.episode_start = np.array(
                data["episode_start"][:n], dtype=np.bool_, copy=True
            )

            mmap_path = path.with_name(path.stem + "_images.mmap")
            # Equal file size does not imply equal data after online checkpointing.
            # Always rebuild atomically (release existing mappings before reload
            # on Windows, where mapped files cannot be replaced).
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mmap", delete=False) as temp:
                temporary_map = Path(temp.name)
            mm = np.memmap(temporary_map, dtype=np.uint8, mode="w+", shape=(n, *image_shape))
            mm[:] = imgs[:n]
            mm.flush()
            del mm
            os.replace(temporary_map, mmap_path)
            buf.images = np.memmap(mmap_path, dtype=np.uint8, mode="r", shape=(n, *image_shape))

            if "term_idx" in data and len(data["term_idx"]):
                for k, img, st in zip(
                    data["term_idx"], data["term_images"], data["term_states"]
                ):
                    ki = int(k)
                    if 0 <= ki < n:
                        buf._term_image[ki] = np.asarray(img, dtype=np.uint8).copy()
                        buf._term_state[ki] = np.asarray(st, dtype=np.float32).copy()

            saved_full = bool(data["full"][0]) if "full" in data else False
            buf.idx = int(data["idx"][0]) % n if saved_full else 0
            buf.full = True
            buf._mmap_path = mmap_path
        return buf

    def summary(self) -> dict[str, Any]:
        return {
            "size": len(self),
            "capacity": self.capacity,
            "sequence_length": self.sequence_length,
            "valid_sequences": int(len(self._valid_start_indices())),
        }
