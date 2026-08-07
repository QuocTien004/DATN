from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        # Sparse terminal next-obs (only when done=True)
        self._term_image: dict[int, np.ndarray] = {}
        self._term_state: dict[int, np.ndarray] = {}

        self.idx = 0
        self.full = False
        self.episode_start = np.ones((capacity,), dtype=np.bool_)
        self._valid_starts_cache: np.ndarray | None = None
        self._valid_starts_key: tuple[int, int, bool, int] | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, transition: Transition, *, is_first: bool = False) -> None:
        i = self.idx
        # Overwriting a slot: drop sparse terminal cache for that index
        self._term_image.pop(i, None)
        self._term_state.pop(i, None)

        self.images[i] = transition.image
        self.states[i] = transition.state
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.dones[i] = transition.done
        if transition.done:
            self._term_image[i] = np.asarray(transition.next_image, dtype=np.uint8).copy()
            self._term_state[i] = np.asarray(transition.next_state, dtype=np.float32).copy()
        self.episode_start[i] = is_first

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
        self._valid_starts_cache = None

    def _valid_start_indices(self) -> np.ndarray:
        """
        Starts for sequences of length L that do not cross episode boundaries
        and have a following frame for next_obs (or terminal next when done).
        """
        n = len(self)
        L = self.sequence_length
        key = (n, L, self.full, self.idx)
        if self._valid_starts_cache is not None and self._valid_starts_key == key:
            return self._valid_starts_cache

        if n < L:
            starts = np.array([], dtype=np.int64)
        else:
            max_start = n - L + 1
            if self.full:
                max_start = min(max_start, self.capacity - L + 1)
            d = self.dones[:n].astype(np.int32)
            prefix = np.concatenate([[0], np.cumsum(d)])
            mid_done = prefix[L - 1 : L - 1 + max_start] > prefix[:max_start]
            last = np.arange(max_start) + (L - 1)
            last_done = d[last].astype(bool)
            ok_next = last_done | ((last + 1) < n)
            if self.full:
                ok_next = last_done | ((last + 1) < self.capacity)
            starts = np.flatnonzero((~mid_done) & ok_next).astype(np.int64)

        self._valid_starts_cache = starts
        self._valid_starts_key = key
        return starts

    def _next_image_at(self, t: int) -> np.ndarray:
        if bool(self.dones[t]):
            return self._term_image.get(t, self.images[t])
        return self.images[t + 1]

    def _next_state_at(self, t: int) -> np.ndarray:
        if bool(self.dones[t]):
            return self._term_state.get(t, self.states[t])
        return self.states[t + 1]

    def sample(
        self,
        batch_size: int,
        *,
        include_next: bool = False,
    ) -> dict[str, np.ndarray]:
        starts = self._valid_start_indices()
        if len(starts) == 0:
            raise RuntimeError(
                "Not enough contiguous sequences in buffer. Collect more data."
            )
        chosen = np.random.choice(starts, size=batch_size, replace=len(starts) < batch_size)
        L = self.sequence_length

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
        }
        # next_* copies many 256x256 frames; WM train does not need it
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

    def save(self, path: str | Path) -> Path:
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

        np.savez_compressed(
            path,
            images=self.images[:n] if not self.full else self.images,
            states=self.states[:n] if not self.full else self.states,
            actions=self.actions[:n] if not self.full else self.actions,
            rewards=self.rewards[:n] if not self.full else self.rewards,
            dones=self.dones[:n] if not self.full else self.dones,
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
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """
        Load bootstrap buffer. Capacity = actual size; images live in a disk
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
            buf.episode_start = np.array(
                data["episode_start"][:n], dtype=np.bool_, copy=True
            )

            mmap_path = path.with_name(path.stem + "_images.mmap")
            expected = n * int(np.prod(image_shape))
            if mmap_path.exists() and mmap_path.stat().st_size == expected:
                buf.images = np.memmap(
                    mmap_path, dtype=np.uint8, mode="r", shape=(n, *image_shape)
                )
            else:
                mm = np.memmap(
                    mmap_path, dtype=np.uint8, mode="w+", shape=(n, *image_shape)
                )
                mm[:] = imgs[:n]
                mm.flush()
                del mm
                buf.images = np.memmap(
                    mmap_path, dtype=np.uint8, mode="r", shape=(n, *image_shape)
                )

            if "term_idx" in data and len(data["term_idx"]):
                for k, img, st in zip(
                    data["term_idx"], data["term_images"], data["term_states"]
                ):
                    ki = int(k)
                    if 0 <= ki < n:
                        buf._term_image[ki] = np.asarray(img, dtype=np.uint8).copy()
                        buf._term_state[ki] = np.asarray(st, dtype=np.float32).copy()

            buf.idx = 0
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
