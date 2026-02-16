"""
RTSP Stream Utilities
Helper functions for RTSP URL management
"""

import re
from typing import Optional, Tuple


def validate_stream_name(name: str) -> bool:
    """
    Validate stream name for MediaMTX compatibility

    MediaMTX stream names should:
    - Be alphanumeric with underscores, hyphens, or periods
    - Not start with special characters
    - Be URL-safe

    Args:
        name: Stream name to validate

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> validate_stream_name("kluis_line")
        True
        >>> validate_stream_name("after-hour")
        True
        >>> validate_stream_name("config/test")  # Contains /
        False
    """
    if not name or not isinstance(name, str):
        return False

    # Allow alphanumeric, underscore, hyphen, period
    # Must start with alphanumeric
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$"
    return bool(re.match(pattern, name))


def build_rtsp_url(server: str, stream_name: str, port: Optional[int] = None) -> str:
    """
    Build RTSP URL from server and stream name

    Args:
        server: RTSP server address (e.g., 'localhost', '192.168.1.100')
        stream_name: Name of the stream (should be config name)
        port: Optional port number (default: 554 for RTSP, 6554 for MediaMTX)

    Returns:
        str: Complete RTSP URL

    Example:
        >>> build_rtsp_url('localhost', 'kluis_line', 6554)
        'rtsp://localhost:6554/kluis_line'

        >>> build_rtsp_url('192.168.1.100', 'after-hour')
        'rtsp://192.168.1.100:554/after-hour'
    """
    if not validate_stream_name(stream_name):
        raise ValueError(
            f"Invalid stream name: '{stream_name}'. "
            "Use only alphanumeric, underscore, hyphen, or period."
        )

    # Clean server string
    server = server.strip()
    if server.startswith("rtsp://"):
        server = server[7:]

    # Remove trailing slashes
    server = server.rstrip("/")

    # Build URL
    if port:
        url = f"rtsp://{server}:{port}/{stream_name}"
    else:
        # Check if server already has port
        if ":" in server:
            url = f"rtsp://{server}/{stream_name}"
        else:
            # Default RTSP port
            url = f"rtsp://{server}:554/{stream_name}"

    return url


def parse_rtsp_url(url: str) -> Tuple[str, int, str]:
    """
    Parse RTSP URL into components

    Args:
        url: RTSP URL to parse

    Returns:
        tuple: (server, port, stream_name)

    Example:
        >>> parse_rtsp_url('rtsp://localhost:6554/kluis_line')
        ('localhost', 6554, 'kluis_line')

        >>> parse_rtsp_url('rtsp://192.168.1.100/stream')
        ('192.168.1.100', 554, 'stream')

    Raises:
        ValueError: If URL format is invalid
    """
    if not url.startswith("rtsp://"):
        raise ValueError(f"Invalid RTSP URL: must start with 'rtsp://' (got: {url})")

    # Remove rtsp:// prefix
    url_body = url[7:]

    # Split by /
    parts = url_body.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid RTSP URL: missing stream path (got: {url})")

    server_part, stream_name = parts

    # Parse server and port
    if ":" in server_part:
        server, port_str = server_part.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")
    else:
        server = server_part
        port = 554  # Default RTSP port

    return (server, port, stream_name)


def get_stream_name_from_config(config_path: str) -> str:
    """
    Extract stream name from config file path

    Args:
        config_path: Path to config YAML file

    Returns:
        str: Stream name (filename without extension)

    Example:
        >>> get_stream_name_from_config('/app/configs/kluis_line.yaml')
        'kluis_line'

        >>> get_stream_name_from_config('configs/after-hour.yaml')
        'after-hour'
    """
    import os

    filename = os.path.basename(config_path)
    # Remove extension
    name = os.path.splitext(filename)[0]

    if not validate_stream_name(name):
        raise ValueError(
            f"Config filename '{name}' is not a valid stream name. "
            "Use only alphanumeric, underscore, hyphen, or period."
        )

    return name


def build_mediamtx_url(stream_name: str, host: str = "localhost", port: int = 6554) -> str:
    """
    Convenience function to build MediaMTX RTSP URL from config name

    Args:
        stream_name: Name of the stream (config name)
        host: MediaMTX server host (default: localhost)
        port: MediaMTX RTSP port (default: 6554)

    Returns:
        str: Complete RTSP URL for MediaMTX

    Example:
        >>> build_mediamtx_url('kluis_line')
        'rtsp://localhost:6554/kluis_line'

        >>> build_mediamtx_url('after-hour', host='192.168.1.10', port=8554)
        'rtsp://192.168.1.10:8554/after-hour'
    """
    return build_rtsp_url(host, stream_name, port)
