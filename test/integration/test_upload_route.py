import pytest
import io


def test_upload_success_returns_session_indexed(
    client, clear_sessions, stub_ingestor, tmp_dirs
):
    files = {"files": ("note.txt", io.BytesIO(b"Hello World"), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is True
    assert data["session_id"]


def test_upload_no_files_validation_error(client, clear_sessions, stub_ingestor):
    response = client.post("/upload", files=[])
    assert response.status_code == 422


def test_upload_ingestor_failure_returns_500(
    client, clear_sessions, stub_ingestor, monkeypatch
):
    import multi_doc_chat.src.document_ingestion.document_ingestion as di
    import main

    class Boom:
        def __init__(self, *a, **k):
            self.session_id = "sess_test"

        def built_retriever(self, *a, **k):
            from multi_doc_chat.exceptions.custom_exceptions import (
                CustomDocumentException,
            )

            raise CustomDocumentException("boom", None)

    monkeypatch.setattr(di, "ChatIngestor", Boom)
    monkeypatch.setattr(main, "ChatIngestor", Boom)
    files = {"files": ("note.txt", io.BytesIO(b"Hello World"), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 500
    assert "boom" in response.json()["detail"].lower()
