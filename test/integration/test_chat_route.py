import pytest


def test_chat_invalid_session_returns_400(client, clear_sessions, stub_rag):
    body = {"session_id": "", "message": "hi"}
    response = client.post("/chat", json=body)
    assert response.status_code == 400
    assert (
        "Invalid or expired session_id. Re-upload documents."
        in response.json()["detail"]
    )


def test_chat_empty_message_returns_400(client, clear_sessions, stub_rag):
    session_id = "session_test"
    import main

    main.SESSIONS_CHAT_HISTORY[session_id] = []
    body = {"session_id": session_id, "message": " "}
    response = client.post("/chat", json=body)
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]


def test_chat_success_returns_answer_and_appends_history(
    client, clear_sessions, stub_rag
):
    session_id = "session_test"
    import main

    main.SESSIONS_CHAT_HISTORY[session_id] = []
    body = {"session_id": session_id, "message": "Hello"}
    response = client.post("/chat", json=body)
    assert response.status_code == 200
    assert response.json()["answer"] == "stubbed answer"
    assert len(main.SESSIONS_CHAT_HISTORY[session_id]) == 2  


def test_chat_failure_returns_500(client, clear_sessions, monkeypatch):
    import pathlib

    pathlib.Path("faiss_index/session_test").mkdir(parents=True, exist_ok=True)

    sess_id = "session_test"
    import main

    main.SESSIONS_CHAT_HISTORY[sess_id] = []

    class BoomRAG:
        def __init__(self, session_id=None):
            pass

        def load_retriever_from_FAISS(self, faiss_sub_dir=None, **kwargs):
            return None

        def llm_invoke(self, message, chat_history):
            from multi_doc_chat.exceptions.custom_exceptions import (
                CustomDocumentException,
            )

            raise CustomDocumentException("chat failed", None)

    monkeypatch.setattr(main, "ConversationalRAG", BoomRAG)
    response = client.post("/chat", json={"session_id": sess_id, "message": "hi"})
    assert response.status_code == 500
    assert "chat failed" in response.json()["detail"].lower()
