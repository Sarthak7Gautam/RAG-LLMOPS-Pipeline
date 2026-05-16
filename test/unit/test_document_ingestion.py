import pathlib
from langchain_core.documents import Document

from multi_doc_chat.src.document_ingestion.document_ingestion import (
    ChatIngestor,
    FaissManager,
    generate_session_id,
)


def test_generate_session_id_format_and_uniqueness():
    a = generate_session_id()
    b = generate_session_id()
    assert a != b
    assert a.startswith("session_") and b.startswith("session_")
    assert (
        len(a.split("_")) == 3
    )  # error maybe here check your actual code to see what session_id generates


def test_chat_ingestor_resolve_dir_use_sessions_dir(tmp_dir, stub_model_loader):
    ing = ChatIngestor(
        user_uploaded_data_temp_base="data",
        faiss_vector_store_base="faiss_index",
        use_sessions_dirs=True,
    )
    assert ing.session_id
    assert str(ing.temp_sub_dir).endswith(ing.session_id)
    assert str(ing.faiss_sub_dir).endswith(ing.session_id)


def test_split_chunks_respect_size_and_overlap(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(
        user_uploaded_data_temp_base="data",
        faiss_vector_store_base="faiss_index",
        use_sessions_dirs=True,
    )
    docs = [Document(page_content="A" * 1200, meta_data={"source": "x.txt"})]
    chunks = ing._split(docs=docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    assert len(docs[0].page_content) <= 500


def test_faiss_manager_add_documents_idempotent(tmp_dirs, stub_model_loader):
    fm = FaissManager(index_dir=pathlib.Path("faiss_index/test"))
    fm.load_or_create_vector_store()  # maybe some error may occur here check here
    docs = [Document(page_content="hello", meta_data={"source": "a"})]
    first = fm.add_documents_in_vector_store(docs)
    second = fm.add_documents_in_vector_store(docs)
    assert first >= 0
    assert second == 0
