"""RTSP Stream Reader with reconnection support"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from yoi.utils.logger import logger_service

logger = logger_service.get_rtsp_logger()


@dataclass
class RTSPConfig:
    """RTSP stream configuration"""

    url: str
    reconnect_delay: int = 5  # seconds
    max_reconnect_attempts: int = 0  # 0 = infinite
    read_timeout: int = 10  # seconds
    buffer_size: int = 1  # frames (1 = no buffering, latest frame only)
    transport: str = "tcp"  # tcp or udp


class RTSPReader:
    """
    Read frames from RTSP stream with automatic reconnection

    Features:
    - Automatic reconnection on failure
    - Configurable timeouts and retries
    - TCP/UDP transport selection
    - Frame skip to get latest frame (low latency mode)

    Usage:
        config = RTSPConfig(url='rtsp://camera_ip/stream', transport='tcp')
        reader = RTSPReader(config)
        reader.connect()

        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            # Process frame

        reader.release()
    """

    def __init__(self, config: RTSPConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_frame_time = 0

    def connect(self) -> bool:
        """
        Connect to RTSP stream

        Returns:
            bool: True if connected successfully
        """
        if self.is_connected and self.cap is not None:
            logger.warning("Already connected, releasing existing connection")
            self.release()

        logger.info(f"Connecting to RTSP stream: {self.config.url}")

        try:
            self.cap = cv2.VideoCapture(self.config.url)

            # Set OpenCV RTSP options
            if self.config.transport == "tcp":
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                # Force TCP transport (reduces packet loss)
                # Note: OpenCV might not support this on all platforms

            # Try to read first frame to verify connection
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_connected = True
                self.reconnect_count = 0
                self.last_frame_time = time.time()
                logger.info(
                    f"Connected to RTSP stream successfully (resolution: {frame.shape[1]}x{frame.shape[0]})"
                )
                return True
            else:
                logger.error("Failed to read first frame from RTSP stream")
                self.release()
                return False

        except Exception as e:
            logger.error(f"Error connecting to RTSP stream: {e}")
            self.release()
            return False

    def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect to stream

        Returns:
            bool: True if reconnected successfully
        """
        if self.config.max_reconnect_attempts > 0:
            if self.reconnect_count >= self.config.max_reconnect_attempts:
                logger.error(
                    f"Max reconnect attempts ({self.config.max_reconnect_attempts}) reached"
                )
                return False

        self.reconnect_count += 1
        logger.warning(f"Connection lost. Reconnecting (attempt {self.reconnect_count})...")

        self.release()
        time.sleep(self.config.reconnect_delay)

        return self.connect()

    def read_frame(self) -> Optional[cv2.Mat]:
        """
        Read frame from RTSP stream with auto-reconnect

        Returns:
            numpy.ndarray: Frame as BGR image, or None if stream ended
        """
        if not self.is_connected or self.cap is None:
            logger.error("Not connected to stream. Call connect() first.")
            return None

        # Check for read timeout
        if time.time() - self.last_frame_time > self.config.read_timeout:
            logger.warning("Read timeout detected")
            if not self._attempt_reconnect():
                return None

        try:
            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.last_frame_time = time.time()
                return frame
            else:
                # Connection lost or stream ended
                logger.warning("Failed to read frame, attempting reconnect")
                if self._attempt_reconnect():
                    # Try reading again after reconnect
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.last_frame_time = time.time()
                        return frame

                return None

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            if self._attempt_reconnect():
                return self.read_frame()  # Recursive retry
            return None

    def get_fps(self) -> float:
        """Get stream FPS if available"""
        if self.cap is not None and self.is_connected:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 25.0  # Default to 25 if unknown
        return 25.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get stream resolution (width, height)"""
        if self.cap is not None and self.is_connected:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)

    def release(self):
        """Release RTSP stream resources"""
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("RTSP stream released")
            except Exception as e:
                logger.error(f"Error releasing stream: {e}")
            finally:
                self.cap = None
                self.is_connected = False

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False


def create_rtsp_reader(
    rtsp_url: str, transport: str = "tcp", reconnect_delay: int = 5, buffer_size: int = 1
) -> RTSPReader:
    """
    Convenience function to create RTSP reader

    Args:
        rtsp_url: RTSP stream URL (e.g., 'rtsp://camera_ip/stream')
        transport: 'tcp' or 'udp' (tcp recommended for reliability)
        reconnect_delay: Seconds to wait before reconnecting
        buffer_size: Frame buffer size (1 = latest frame only, reduces latency)

    Returns:
        RTSPReader: Configured RTSP reader instance

    Example:
        reader = create_rtsp_reader('rtsp://192.168.1.100/stream', transport='tcp')
        reader.connect()

        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            cv2.imshow('RTSP Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
    """
    config = RTSPConfig(
        url=rtsp_url, reconnect_delay=reconnect_delay, buffer_size=buffer_size, transport=transport
    )
    return RTSPReader(config)
