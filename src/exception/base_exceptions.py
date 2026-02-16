"""
YOI Base Exception Classes
"""


class YOIException(Exception):
    """Base exception for all YOI errors"""

    def __init__(self, message: str, error_code: str = "YOI_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class ConfigException(YOIException):
    """Exception for configuration-related errors"""

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR"):
        super().__init__(message, error_code)


class EngineException(YOIException):
    """Exception for vision engine errors"""

    def __init__(self, message: str, error_code: str = "ENGINE_ERROR"):
        super().__init__(message, error_code)


class ValidationException(YOIException):
    """Exception for validation errors"""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message, error_code)


class ProcessingException(YOIException):
    """Exception for processing pipeline errors"""

    def __init__(self, message: str, error_code: str = "PROCESSING_ERROR"):
        super().__init__(message, error_code)
