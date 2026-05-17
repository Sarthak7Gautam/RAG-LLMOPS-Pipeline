import pytest

from multi_doc_chat.src.document_pipeline.document_retriever import ConversationalRAG
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException


def test_conversationalrag_error_handling(tmp_dirs, stub_model_loader):
    with pytest.raises(CustomDocumentException):
        ConversationalRAG(session_id="s1")

    with pytest.raises(CustomDocumentException):
        ConversationalRAG(session_id="no_path")
