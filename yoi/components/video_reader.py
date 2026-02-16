"""Video readers for files and RTSP streams."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from yoi.utils.logger import logger_service


class BaseVideoReader(ABC):
    """Abstract base class for video readers."""

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from source."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return source FPS."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Return total frame count (-1 when unknown, e.g. RTSP)."""
        pass

    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)"""
        pass

    @abstractmethod
    def close(self):
        """Close reader"""
        pass


class FileVideoReader(BaseVideoReader):
    """Video reader for local video files."""

    def __init__(self, file_path: str, max_fps: Optional[int] = None):
        self.file_path = Path(file_path)
        self.max_fps = max_fps
        self.logger = logger_service.get_video_logger()

        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self.cap = cv2.VideoCapture(str(self.file_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {file_path}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        self.logger.info(
            f"VideoReader initialized: {self.file_path.name} "
            f"({self._width}x{self._height}) @ {self._fps} FPS"
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from file source."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
        return ret, frame

    def get_fps(self) -> float:
        return self._fps

    def get_current_frame_idx(self) -> int:
        return self.current_frame_idx

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def close(self):
        if self.cap:
            self.cap.release()

    def rewind(self) -> bool:
        """Rewind video file to first frame for looping playback."""
        if not self.cap:
            return False
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ok:
            self.current_frame_idx = 0
            self.logger.info(f"VideoReader rewound: {self.file_path.name}")
        return bool(ok)


class RTSPVideoReader(BaseVideoReader):
    """Video reader for RTSP streams."""

    def __init__(self, rtsp_url: str, max_fps: Optional[int] = None, buffer_size: int = 1):
        self.rtsp_url = rtsp_url
        self.max_fps = max_fps
        self.buffer_size = buffer_size
        self.logger = logger_service.get_rtsp_logger()

        if not rtsp_url.startswith("rtsp://"):
            raise ValueError(f"Invalid RTSP URL: {rtsp_url}")

        self.logger.info(f"RTSP CONNECTING: {rtsp_url} (buffer_size={buffer_size})")

        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        if not self.cap.isOpened():
            self.logger.error(f"RTSP CONNECT FAILED: {rtsp_url}")
            raise RuntimeError(f"Cannot connect to RTSP: {rtsp_url}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        self.last_read_time = time.time()

        self.logger.info(
            f"RTSP CONNECTED: {rtsp_url} ({self._width}x{self._height}) @ {self._fps} FPS"
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from RTSP stream."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
            self.last_read_time = time.time()
        else:
            self.logger.warning("Failed to read frame, attempting reconnect...")
            self._reconnect()
            ret, frame = self.cap.read()

        return ret, frame

    def _reconnect(self):
        """Reconnect to RTSP stream."""
        self.logger.info("RTSP RECONNECT: releasing and reopening stream")
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def get_fps(self) -> float:
        return self._fps

    def get_current_frame_idx(self) -> int:
        return self.current_frame_idx

    def get_frame_count(self) -> int:
        return -1

    def get_frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def close(self):
        if self.cap:
            self.cap.release()


class VideoReader:
    """Factory class for video readers."""

    @staticmethod
    def create(source: str, max_fps: Optional[int] = None, buffer_size: int = 1) -> BaseVideoReader:
        """
        Create a video reader based on source type.

        Args:
            source: File path or RTSP URL
            max_fps: Max FPS for processing
            buffer_size: Buffer size for RTSP

        Returns:
            Appropriate video reader instance
        """
        if source.startswith("rtsp://"):
            return RTSPVideoReader(source, max_fps, buffer_size)
        else:
            return FileVideoReader(source, max_fps)

    @staticmethod
    def create_frame_generator(
        reader: BaseVideoReader, max_fps: Optional[int] = None, loop_file: bool = False
    ) -> Generator:
        """
        Create a frame generator with optional FPS limiting.

        Args:
            reader: Video reader instance
            max_fps: Max FPS for processing

        Yields:
            Tuple of (frame_idx, frame)
        """
        frame_delay = 1.0 / max_fps if max_fps else 0
        last_frame_time = time.time()

        frame_idx = 0
        while True:
            ret, frame = reader.read_frame()
            if not ret:
                if loop_file and isinstance(reader, FileVideoReader) and reader.rewind():
                    continue
                break

            if frame_delay > 0:
                elapsed = time.time() - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            yield frame_idx, frame
            frame_idx += 1
            last_frame_time = time.time()
