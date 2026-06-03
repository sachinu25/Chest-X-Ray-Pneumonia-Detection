"""
Custom exception handler for the X-ray classification project.

Provides detailed error messages including filename, line number, and
traceback information for debugging.
"""

import os
import sys
import traceback


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Build a detailed error message with file, line, and traceback info.

    Handles cases where traceback information may not be available
    (e.g., when called outside an exception context).
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error in [{file_name}] at line [{line_number}]: {str(error)}"
        )
    else:
        error_message = f"Error: {str(error)}"

    return error_message


class XRayException(Exception):
    """
    Custom exception that captures file name, line number, and error details.

    Usage:
        try:
            ...
        except Exception as e:
            raise XRayException(e, sys)
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return f"XRayException('{self.error_message}')"
