from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from tqdm.auto import tqdm

from ddmtools.utils.number import get_centre_matrix

Frame = np.ndarray
ImageDimension = Tuple[int, int]


class FrameAccessor(ABC):
    """
    Obtains a frame in a caching manner.
    """

    def __init__(self) -> None:
        self.frame: Optional[Frame] = None

    def get(self, crop_target: Optional[ImageDimension] = None) -> Frame:
        if self.frame is None:
            self.load()

        assert self.frame is not None

        if crop_target is not None and self.frame.shape != crop_target:
            self.crop(crop_target=crop_target)

        return self.frame

    def unload(self) -> None:
        self.frame = None

    @abstractmethod
    def load(self) -> None:
        ...

    def crop(self, crop_target: ImageDimension) -> None:
        self.frame = get_centre_matrix(self.get(), crop_target)

    def display(self) -> Figure:
        fig = plt.figure(dpi=150)
        plt.imshow(self.get(), plt.cm.gray)

        return fig


class ImageFrameAccessor(FrameAccessor):
    def __init__(self, img_path: Path):
        super().__init__()

        self.img_path = img_path

    def load(self) -> None:
        self.frame = np.asarray(Image.open(self.img_path)).astype(float)


class VideoFrameAccessor(FrameAccessor):
    def __init__(self, video: cv2.VideoCapture, frame_num: int):
        super().__init__()

        self.video = video
        self.frame_num = frame_num

    def load(self) -> None:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)

        success, frame_rgb = self.video.read()

        if not success:
            raise IOError("Empty frame.")

        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        self.frame = frame.astype(float)


class Framestack:
    def __init__(self, frames: Sequence[FrameAccessor], shape: Optional[ImageDimension] = None):
        self.frames = list(frames)
        self.current_frame_idx = 0

        self._shape: Optional[ImageDimension] = shape

    def load(self, show_progress: bool = True) -> None:
        for frame in tqdm(list(self.frames), disable=not show_progress):
            try:
                frame.load()
            except IOError:
                self.frames.remove(frame)

    def unload(self) -> None:
        for frame in self.frames:
            frame.unload()

    @classmethod
    def from_folder(
        cls, folder: Path, glob_pattern: str = "*.pgm", crop_target: Optional[ImageDimension] = None
    ) -> Framestack:
        paths = sorted(folder.glob(glob_pattern))
        accessors = [ImageFrameAccessor(path) for path in paths]

        return cls(accessors, shape=crop_target)

    @classmethod
    def from_video(cls, path: Path, crop_target: Optional[ImageDimension] = None) -> Framestack:
        video = cv2.VideoCapture(str(path))

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        accessors = [VideoFrameAccessor(video, i) for i in range(frame_count)]

        return cls(accessors, shape=crop_target)

    def __len__(self) -> int:
        return len(self.frames)

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
        return self.frames[idx].get(crop_target=self._shape)

    @property
    def shape(self) -> ImageDimension:
        return self._shape or self[0].shape

    @shape.setter
    def shape(self, value: ImageDimension) -> None:
        self._shape = value
