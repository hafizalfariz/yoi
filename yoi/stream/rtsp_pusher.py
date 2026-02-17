"""RTSP Stream Pusher for MediaMTX server.

Push processed video frames to RTSP server for viewing.
"""

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from yoi.utils.logger import logger_service

logger = logger_service.get_rtsp_logger()


@dataclass
class RTSPPushConfig:
    """RTSP push configuration for MediaMTX"""

    server_url: str  # e.g., 'rtsp://localhost:6554/bag'
    fps: int = 25
    width: int = 1920
    height: int = 1080
    bitrate: str = "2M"  # FFmpeg bitrate (e.g., '2M', '4M', '8M')
    codec: str = "libx264"  # Video codec
    preset: str = "ultrafast"  # FFmpeg preset (ultrafast, fast, medium)
    pix_fmt: str = "yuv420p"  # Pixel format
    rtsp_transport: str = "tcp"  # tcp or udp


class RTSPPusher:
    """
    Push video frames to MediaMTX RTSP server using FFmpeg

    Features:
    - Real-time H.264 encoding
    - Configurable quality/performance
    - Automatic process management
    - Error handling and recovery

    Usage:
        config = RTSPPushConfig(
            server_url='rtsp://localhost:6554/processed',
            fps=25,
            width=1920,
            height=1080
        )
        pusher = RTSPPusher(config)
        pusher.start()

        # Push frames in loop
        for frame in video_frames:
            if not pusher.push_frame(frame):
                break

        pusher.stop()
    """

    def __init__(self, config: RTSPPushConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.frame_count = 0
        self._stderr_thread: Optional[threading.Thread] = None
        self.last_startup_output: List[str] = []
        self._max_startup_lines = 200

    def _build_ffmpeg_command(self) -> list:
        """
        Build FFmpeg command for RTSP push

        Returns:
            list: FFmpeg command as list of arguments
        """
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",  # OpenCV uses BGR
            "-s",
            f"{self.config.width}x{self.config.height}",
            "-use_wallclock_as_timestamps",
            "1",
            "-fflags",
            "+genpts",
            "-r",
            str(self.config.fps),
            "-i",
            "-",  # Input from stdin
            "-an",
            "-c:v",
            self.config.codec,
            "-preset",
            self.config.preset,
            "-tune",
            "zerolatency",
            "-b:v",
            self.config.bitrate,
            "-pix_fmt",
            self.config.pix_fmt,
            "-g",
            str(self.config.fps * 2),  # GOP size = 2 seconds
            "-f",
            "rtsp",
            "-rtsp_transport",
            self.config.rtsp_transport,
            "-rtsp_flags",
            "prefer_tcp",
            "-rw_timeout",
            "30000000",
            "-muxdelay",
            "0.1",
            "-muxpreload",
            "0",
            self.config.server_url,
        ]

        return command

    def start(self) -> bool:
        """
        Start FFmpeg process for RTSP streaming

        Returns:
            bool: True if started successfully
        """
        if self.is_running and self.process is not None:
            logger.warning("RTSP pusher already running")
            return True

        max_attempts = 5
        startup_probe_seconds = 1.0

        for attempt in range(1, max_attempts + 1):
            self.last_startup_output = []
            try:
                command = self._build_ffmpeg_command()
                logger.info(
                    f"Starting RTSP push to: {self.config.server_url} "
                    f"(attempt {attempt}/{max_attempts})"
                )
                logger.debug(f"FFmpeg command: {' '.join(command)}")

                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=10**8,  # Large buffer for smooth streaming
                )

                # Start background thread to capture stderr lines from FFmpeg
                def _stderr_reader(proc: subprocess.Popen):
                    try:
                        if not proc or not proc.stderr:
                            return
                        while True:
                            line = proc.stderr.readline()
                            if not line:
                                # EOF
                                break
                            try:
                                decoded = line.decode("utf-8", errors="replace").rstrip()
                            except Exception:
                                decoded = str(line)
                            # Store limited recent startup output
                            if len(self.last_startup_output) < self._max_startup_lines:
                                self.last_startup_output.append(decoded)
                            logger.debug(f"FFmpeg: {decoded}")
                    except Exception as e:
                        logger.debug(f"Error reading FFmpeg stderr: {e}")

                self._stderr_thread = threading.Thread(
                    target=_stderr_reader, args=(self.process,), daemon=True
                )
                self._stderr_thread.start()

                # Give FFmpeg a moment; if it exits early, startup failed.
                time.sleep(startup_probe_seconds)
                return_code = self.process.poll()
                if return_code is not None:
                    logger.warning(
                        f"FFmpeg exited during RTSP startup with code {return_code} "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    if self.last_startup_output:
                        for line in self.last_startup_output[-10:]:
                            logger.warning(f"FFmpeg startup: {line}")

                    self.process = None
                    self.is_running = False

                    if attempt < max_attempts:
                        time.sleep(1.0)
                        continue

                    return False

                self.is_running = True
                self.frame_count = 0
                logger.info("RTSP pusher started successfully")
                return True

            except FileNotFoundError:
                logger.error("FFmpeg not found. Please install FFmpeg and add to PATH")
                return False
            except Exception as e:
                logger.error(f"Error starting RTSP pusher: {e}")
                self.process = None
                self.is_running = False
                if attempt < max_attempts:
                    time.sleep(1.0)
                    continue
                return False

        return False

    def push_frame(self, frame: np.ndarray) -> bool:
        """
        Push a single frame to RTSP stream

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            bool: True if frame pushed successfully
        """
        if not self.is_running or self.process is None:
            logger.error("RTSP pusher not running. Call start() first.")
            return False

        # Check if process is still alive
        if self.process.poll() is not None:
            logger.error("FFmpeg process died unexpectedly")
            self.is_running = False
            return False

        try:
            # Resize frame if dimensions don't match config
            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(frame, (self.config.width, self.config.height))

            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            # Optional immediate flush; can reduce latency but may lower throughput.
            flush_env = os.getenv("YOI_RTSP_FLUSH_EVERY_FRAME", "0").strip().lower()
            flush_every_frame = flush_env in {"1", "true", "on", "yes"}
            if flush_every_frame:
                try:
                    self.process.stdin.flush()
                except Exception:
                    pass
            self.frame_count += 1

            # Log early progress so we can see whether frames are being delivered
            if self.frame_count <= 5:
                logger.info(f"Pushed initial frame {self.frame_count} to RTSP")
            elif self.frame_count % 100 == 0:
                logger.debug(f"Pushed {self.frame_count} frames to RTSP")

            return True

        except BrokenPipeError:
            logger.error("FFmpeg pipe broken (stream disconnected)")
            self.is_running = False
            return False
        except Exception as e:
            logger.error(f"Error pushing frame: {e}")
            return False

    def stop(self):
        """Stop FFmpeg process and cleanup"""
        if self.process is not None:
            try:
                logger.info(f"Stopping RTSP pusher (pushed {self.frame_count} frames)")

                # Close stdin to signal end of stream
                if self.process.stdin:
                    try:
                        self.process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass

                # Wait for process to finish (with timeout)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg didn't stop gracefully, terminating")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()

                logger.info("RTSP pusher stopped")

            except Exception as e:
                logger.error(f"Error stopping RTSP pusher: {e}")
            finally:
                self.process = None
                self.is_running = False

    def restart(self) -> bool:
        """
        Restart FFmpeg process (useful for error recovery)

        Returns:
            bool: True if restarted successfully
        """
        logger.info("Restarting RTSP pusher")
        self.stop()
        return self.start()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


def create_rtsp_pusher(
    server_url: str,
    fps: int = 25,
    width: int = 1920,
    height: int = 1080,
    bitrate: str = "2M",
    preset: str = "ultrafast",
) -> RTSPPusher:
    """
    Convenience function to create RTSP pusher for MediaMTX

    Args:
        server_url: MediaMTX RTSP URL (e.g., 'rtsp://localhost:6554/processed')
        fps: Stream frame rate
        width: Video width in pixels
        height: Video height in pixels
        bitrate: FFmpeg bitrate (e.g., '2M', '4M', '8M')
        preset: FFmpeg encoding preset ('ultrafast', 'fast', 'medium')

    Returns:
        RTSPPusher: Configured RTSP pusher instance

    Example:
        # Create pusher for MediaMTX server
        pusher = create_rtsp_pusher(
            server_url='rtsp://localhost:6554/bag',
            fps=25,
            width=1920,
            height=1080,
            bitrate='4M',
            preset='ultrafast'
        )

        pusher.start()

        # Process video with inference
        cap = cv2.VideoCapture('input.mp4')
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Your inference code here
            processed_frame = your_inference(frame)

            # Push to RTSP for viewing
            pusher.push_frame(processed_frame)

        cap.release()
        pusher.stop()
    """
    config = RTSPPushConfig(
        server_url=server_url, fps=fps, width=width, height=height, bitrate=bitrate, preset=preset
    )
    return RTSPPusher(config)
