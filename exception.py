import sys
from logger import logging

def error_message_detail(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"error occured in python file [{filename}] at line#{exc_tb.tb_lineno} error message '[{str(error)}'] "
    
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self)->str:
        return self.error_message


# if __name__ == '__main__':
#     try:
#         a= 1/0
#     except Exception as e:
#         logging.info('Divided by Zero')
#         raise CustomException(e,sys)
