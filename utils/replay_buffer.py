from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Transition:
    """Single env step stored in the replay buffer."""

    image: np.ndarray       # uint8 (H, W, C) or (C, H, W)
    state: np.ndarray       # float32 vector
    action: np.ndarray      # float32
    reward: float
    done: bool
    next_image: np.ndarray
    next_state: np.ndarray


class ReplayBuffer:
    """
    Sequence-capable replay buffer for World Model training.

    Stores transitions from MetaDrive. Sampling returns contiguous
    sequences of length `sequence_length` (required for GRU / RSSM).

    Storage uses NumPy arrays in RAM. For larger runs, dump to
    `save_dir` periodically via `save()` / `load()`.
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
        self.next_images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        self.idx = 0
        self.full = False
        # episode_start[i] = True if index i starts a new episode (after done)
        self.episode_start = np.ones((capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, transition: Transition, *, is_first: bool = False) -> None:
        i = self.idx
        self.images[i] = transition.image
        self.states[i] = transition.state
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.dones[i] = transition.done
        self.next_images[i] = transition.next_image
        self.next_states[i] = transition.next_state
        self.episode_start[i] = is_first

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def _valid_start_indices(self) -> np.ndarray:
        """Indices where a full contiguous sequence fits without crossing episode ends mid-way naively.

        Simple rule: start at `t` if none of dones[t : t+L-1] is True
        (sequence may end with done at last step).
        """
        n = len(self)
        L = self.sequence_length
        if n < L:
            return np.array([], dtype=np.int64)

        valid = []
        for start in range(n - L + 1):
            # Disallow wrapping around the circular buffer for simplicity (Phase A).
            if self.full and start + L > self.capacity:
                continue
            window_done = self.dones[start : start + L - 1]
            if window_done.any():
                continue
            valid.append(start)
        return np.asarray(valid, dtype=np.int64)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Sample `batch_size` sequences of length `sequence_length`.

        Returns dict of arrays with leading dims (B, T, ...).
        """
        starts = self._valid_start_indices()
        if len(starts) == 0:
            raise RuntimeError(
                "Not enough contiguous sequences in buffer. Collect more data."
            )
        chosen = np.random.choice(starts, size=batch_size, replace=len(starts) < batch_size)
        L = self.sequence_length

        def gather(arr: np.ndarray) -> np.ndarray:
            return np.stack([arr[s : s + L] for s in chosen], axis=0)

        return {
            "image": gather(self.images),
            "state": gather(self.states),
            "action": gather(self.actions),
            "reward": gather(self.rewards),
            "done": gather(self.dones),
            "next_image": gather(self.next_images),
            "next_state": gather(self.next_states),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            images=self.images,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            next_images=self.next_images,
            next_states=self.next_states,
            episode_start=self.episode_start,
            idx=np.array([self.idx]),
            full=np.array([self.full]),
            meta=np.array(
                [self.capacity, self.sequence_length, self.state_dim, self.action_dim],
                dtype=np.int64,
            ),
            image_shape=np.array(self.image_shape, dtype=np.int64),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        capacity, sequence_length, state_dim, action_dim = data["meta"].tolist()
        image_shape = tuple(int(x) for x in data["image_shape"].tolist())
        buf = cls(
            capacity=capacity,
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            sequence_length=sequence_length,
        )
        buf.images[:] = data["images"]
        buf.states[:] = data["states"]
        buf.actions[:] = data["actions"]
        buf.rewards[:] = data["rewards"]
        buf.dones[:] = data["dones"]
        buf.next_images[:] = data["next_images"]
        buf.next_states[:] = data["next_states"]
        buf.episode_start[:] = data["episode_start"]
        buf.idx = int(data["idx"][0])
        buf.full = bool(data["full"][0])
        return buf

    def summary(self) -> dict[str, Any]:
        return {
            "size": len(self),
            "capacity": self.capacity,
            "sequence_length": self.sequence_length,
            "valid_sequences": int(len(self._valid_start_indices())),
        }
