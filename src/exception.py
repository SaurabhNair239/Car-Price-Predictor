import sys
import logging
def err_msg_details(err,err_info:sys):
   _,_,exc_tb = err_info.exc_info() 
   file_name = exc_tb.tb_frame.f_code.co_filename
   line_num = exc_tb.tb_lineno
   
   err_msg = "Error in python script names [{0}] line number [{1}] error message [{2}]".format(
      file_name, 
      line_num,
      str(err) 

   )
   return err_msg


class CustomException(Exception):
      def __init__(self,err_msg,err_detail:sys):
         super().__init__(err_msg)
         self.err_msg = err_msg_details(err_msg,err_info=err_detail)

      def __str__(self):
        return self.err_msg
      



    