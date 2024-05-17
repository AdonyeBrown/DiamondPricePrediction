import sys


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exec_traceback = error_details.exc_info()

        self.lineno = exec_traceback.tb_lineno
        self.filename = exec_traceback.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message)
        )


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        raise CustomException(e, sys)
