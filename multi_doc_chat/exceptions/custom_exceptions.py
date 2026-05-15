
class CustomDocumentException(Exception):
    def __init__(
        self,
        error_message: str ,
        error_details: dict = None,
    ):
        
        super().__init__(error_message)
        self.message = error_message
        self.error_details = error_details or {}

