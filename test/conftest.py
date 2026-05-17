import os
import pathlib
import sys
import pytest

os.environ.setdefault(
    "PYTHONPATH", str(pathlib.Path(__file__).resolve().parents[1] / "multi_doc_chat")
)  # creates a search radar for python because when you run pytest it helps you find the accurate folder multi_doc_chat where the core modules are located
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "fake_provider")

from fastapi.testclient import TestClient

ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
)  # returns the project name like RAG-LLMOPS-Pipeline
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def clear_sessions():
    main.SESSIONS_CHAT_HISTORY.clear()
    yield
    main.SESSIONS_CHAT_HISTORY.clear()


@pytest.fixture
def tmp_dirs(
    tmp_path: pathlib.Path,
):  # tmp_path is a built in fixture that creates a hidden system temporary folder inside the os hover over it to see the details
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(
            tmp_path
        )  # this line changes the python focus to that hidden system temporary folder
        yield {
            "data": data_dir,
            "faiss": faiss_dir,
        }  # this pauses the setup and passes the control to tests functions who call it and finish their operations in that hidden system folder and later returns to this fixture when execution is done
    finally:
        os.chdir(
            cwd
        )  # and finally control reaches back to the project root folder RAG-LLMOPS-Pipeline


class _StubEmbeddings:
    def embed_query(self, text: str):
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.0, 0.1, 0.2] for _ in texts]

    def __call__(self, text: str): # allows you to treat the instance of a object exactly like a regular function and call it directly using parentheses
        """     This is without __call__
                # Creating the object
                model = _StubEmbeddings()

                # Running it requires the method name
                vector = model.embed_query("hello")

                With _call_
                # Creating the object
                model = _StubEmbeddings()

                # You can call the object DIRECTLY like a function!
                vector = model("hello")
        """
    
        return [0.0, 0.1, 0.2]


class _StubLLM:
    def invoke(self, input: str):
        return "stubbed answer"


@pytest.fixture
def stub_model_loader(monkeypatch):
    import multi_doc_chat.utils.model_loader as ml_load
    from multi_doc_chat.utils import model_loader as ml_load2

    class FakeApiKeyManager:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x"}

        def get(self, key: str) -> str:
            return self.api_keys[key]

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyManager()
            self.config = {
                "embedding_model": {"model_name": "fake_embed"},
                "llm": {
                    "groq": {
                        "provider": "groq",
                        "model_name": "fake-llm",
                        "temperature": 0.0,
                        "max_output_tokens": 128,
                    }
                },
            }

        def load_embedding_model(self):
            return _StubEmbeddings()

        def load_llm(self):
            return _StubLLM()

    monkeypatch.setattr(ml_load, "ApiKeyManager", FakeApiKeyManager)
    monkeypatch.setattr(ml_load, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(ml_load2, "ApiKeyManager", FakeApiKeyManager)
    monkeypatch.setattr(ml_load2, "ModelLoader", FakeModelLoader)

    import multi_doc_chat.src.document_ingestion.document_ingestion as di
    import multi_doc_chat.src.document_pipeline.document_retriever as dr

    monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(dr, "ModelLoader", FakeModelLoader)
    yield FakeModelLoader


@pytest.fixture
def stub_ingestor(monkeypatch):
    import multi_doc_chat.src.document_ingestion.document_ingestion as di

    class FakeIngestor:
        def __init__(self, use_sessions_dirs=True, **kwargs):
            self.use_sessions_dir = use_sessions_dirs
            self.session_id = "session_test"

        def built_retriever(self, uploaded_files, **kwargs):
            return None

    monkeypatch.setattr(di, "ChatIngestor", FakeIngestor)
    monkeypatch.setattr(main, "ChatIngestor", FakeIngestor)
    yield FakeIngestor


@pytest.fixture
def stub_rag(monkeypatch):
    import multi_doc_chat.src.document_pipeline.document_retriever as dr

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
            self.retriever = retriever

        def load_retriever_from_FAISS(self, faiss_sub_dir, **kwargs):
            return None

        def llm_invoke(self, user_input, chat_history=None):
            return "stubbed answer"

    monkeypatch.setattr(dr, "ConversationalRAG", FakeRAG)
    monkeypatch.setattr(main, "ConversationalRAG", FakeRAG)
    yield FakeRAG
