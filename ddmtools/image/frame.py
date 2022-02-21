from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, cast

import numpy as np

from PIL import Image

from ddmtools.utils import get_centre_matrix

Frame = np.ndarray


class Framestack:
    def __init__(self, frame_paths: list[Path]):
        self.frame_paths = frame_paths
        self.current_frame_idx = 0

        self.frame_cache: list[Union[None, Frame]] = [None] * len(self.frame_paths)

        self.preloaded: bool = False
        self.crop_target: Optional[tuple[int, int]] = None

    def crop(self, new_shape: tuple[int, int]) -> None:
        self.crop_target = new_shape

    def compress(self) -> None:
        """
        Call after having done crop.
        """
        for i, frame in enumerate(self.frame_cache):
            self.frame_cache[i] = frame

    def preload(self) -> None:
        for frame in self:
            pass

        self.preloaded = True

    @classmethod
    def from_folder(cls, folder: Path, glob_pattern: str = "*.pgm") -> Framestack:
        paths = sorted(folder.glob(glob_pattern))

        return cls(paths)

    def __len__(self) -> int:
        return len(self.frame_cache)

    def __iter__(self) -> Framestack:
        return self

    def __next__(self) -> Frame:
        try:
            frame = self[self.current_frame_idx]
            self.current_frame_idx += 1
            return frame
        except IndexError:
            self.current_frame_idx = 0
            raise StopIteration

    def __getitem__(self, idx: int) -> Frame:
        if self.frame_cache[idx] is None:
            frame = self._read_frame(self.frame_paths[idx])
        else:
            frame: np.ndarray = self.frame_cache[idx]

        if self.crop_target is not None and frame.shape != self.crop_target:
            self.frame_cache[idx] = get_centre_matrix(frame, self.crop_target)

        return frame

    @property
    def shape(self) -> tuple[int, int]:
        shape: tuple[int, int] = self[0].shape

        return shape

    def delete_cache(self) -> None:
        self.frame_cache = [None] * len(self.frame_paths)

        self.preloaded = False

    def _read_frame(self, path: Path) -> Frame:
        frame = np.asarray(Image.open(path)).astype(float)

        return frame

    def to_array(self) -> np.ndarray:
        if not self.preloaded:
            self.preload()

        return np.stack(cast(List[Frame], self.frame_cache))
