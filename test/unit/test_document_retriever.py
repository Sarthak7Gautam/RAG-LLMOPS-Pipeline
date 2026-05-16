import pytest

from multi_doc_chat.src.document_pipeline.document_retriever import ConversationalRAG
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException


def test_conversationalrag_error_handling(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(CustomDocumentException):
        rag.invoke("hello")

    with pytest.raises(CustomDocumentException):
        rag.load_retriever_from_FAISS(faiss_sub_dir="faiss_index/no_path")
