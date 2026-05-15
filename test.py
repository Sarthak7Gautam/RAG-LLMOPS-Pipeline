from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from multi_doc_chat.logging.custom_logger import CustomLogger

log = CustomLogger().get_custom_logger()

try:
    1 / 0

except Exception :
    log.exception("code_failed")
    
