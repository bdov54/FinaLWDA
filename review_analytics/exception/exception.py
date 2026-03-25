import sys

class ReviewAnalyticsException(Exception):
    def __init__(self, error_message, error_details=None):
        self.error_message = error_message

        if error_details is None:
            error_details = sys

        _, _, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            self.lineo = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineo = None
            self.file_name = None

    def __str__(self):
        if self.file_name and self.lineo:
            return f"Error in [{self.file_name}] line [{self.lineo}] message [{self.error_message}]"
        return f"Error message [{self.error_message}]"