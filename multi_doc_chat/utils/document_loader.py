from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pathlib import Path
from langchain_core.documents import Document
from typing import List
from multi_doc_chat.logging.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException


logger = CustomLogger()

log = logger.get_custom_logger()


class Load_Documents:
    def load_docs(self, file_path: Path) -> List[Document]:
        try:
            file = Path(file_path)
            loader = None

            suf = file.suffix.lower()

            if suf == ".pdf":
                loader = PyPDFLoader(file_path=file)  # if error occurs make it str(file)
                log.info("PDF File read successfully")

            elif suf == ".docx":
                loader = Docx2txtLoader(file)
                log.info(".docx File read successfully")

            elif suf == ".txt":
                loader = TextLoader(file)
                log.info(".txt File read successfully")

            else:
                log.error(
                    f"Unsupported file extension please upload file from {self.SUPPORTED_EXTENSIONS}"
                )
                return []

            try:
                return loader.load()
            except Exception as e:
                raise CustomDocumentException("Failed to load the file", e)
        except Exception as e:
            raise CustomDocumentException("Unexpected error occured while trying to load the file", str(e))
