"""
Logger Service - Handle all application logging
"""

from datetime import datetime
from typing import List, Literal

from ..models import LogMessage


class LoggerService:
    """Centralized logging service"""

    def __init__(self):
        self.logs: List[LogMessage] = []
        self.max_logs = 100  # Keep last 100 logs

    def log(self, level: Literal["INFO", "WARNING", "ERROR"], category: str, message: str):
        """Add log entry"""
        log_entry = LogMessage(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            level=level,
            category=category,
            message=message,
        )
        self.logs.append(log_entry)

        # Keep only last N logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

        # Print to console
        print(f"[{log_entry.timestamp}] {level:7} [{category}] {message}")

        return log_entry

    def info(self, category: str, message: str):
        """Log INFO level"""
        return self.log("INFO", category, message)

    def warning(self, category: str, message: str):
        """Log WARNING level"""
        return self.log("WARNING", category, message)

    def error(self, category: str, message: str):
        """Log ERROR level"""
        return self.log("ERROR", category, message)

    def get_recent_logs(self, count: int = 50) -> List[LogMessage]:
        """Get recent logs"""
        return self.logs[-count:]

    def clear(self):
        """Clear all logs"""
        self.logs.clear()
        self.info("SYSTEM", "Logs cleared")

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()


# Global logger instance
logger = LoggerService()
