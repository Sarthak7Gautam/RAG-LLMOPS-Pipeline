from fastapi import File, UploadFile, FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from fastapi.exceptions import HTTPException
from multi_doc_chat.logging.custom_logger import CustomLogger
from multi_doc_chat.src.document_ingestion.document_ingestion import ChatIngestor
from multi_doc_chat.src.document_pipeline.document_retriever import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.utils.model_loader import ModelLoader


logger = CustomLogger()

log = logger.get_custom_logger()

## FastAPI Initialization ##

app = FastAPI(title="RAG Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True,
)

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

SESSIONS_CHAT_HISTORY: Dict[str, any] = {}

## Request and Response Classes


class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatAnswer(BaseModel):
    answer: str


## FastAPI FileAdapter


class FastApiFileAdapter:
    "When given a File from FastApi it adds a file adapter and a getbuffer attribute to avoid no attribute found error"

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or uf.file or "name"

    def get_buffer(self) -> bytes:
        self._uf.file.seek(0)
        log.info("get_buffer() function call successful")
        return self._uf.file.read()


## FastAPI EndPoints ##


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    # request parameter is needed because Jinja2 cannot function without it
    return templates.TemplateResponse(
        request=request, name="index.html"
    )  # context is needed to send data from python to html file


@app.post("/upload", response_model=UploadResponse)
async def upload_file(files: List[UploadFile] = File(...)) -> UploadResponse:
    try:
        if not files:
            log.error("No file found in the upload_file function")
            raise HTTPException(status_code=404, detail="Please upload a file")

        wrapped_files = [FastApiFileAdapter(file) for file in files]
        log.info(f"File wrapped success, {len(files)}")

        # file local save, doc_load, split, embeddings_generation, save embeddings in vector_store is done by ChatIngestor
        ingestor = ChatIngestor(use_sessions_dirs=True)
        session_id = ingestor.session_id

        ingestor.built_retriever(
            uploaded_files=wrapped_files,
            k=4,
            search_type="mmr",
            fetch_k=12,
            lambda_mult=0.5,
        )

        SESSIONS_CHAT_HISTORY[session_id] = {}

        log.info("File upload success", session_id=session_id)
        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message="File Upload Successful and Indexing complete with MMR",
        )
    except CustomDocumentException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File Upload Failed : {e}")


@app.post("/chat", response_model=ChatAnswer)
async def chat(request: ChatRequest) -> ChatAnswer:
    try:
        session_id = request.session_id
        message = request.message.strip()

        session_faiss_dir = Path("faiss_index") / session_id

        if not session_id or not session_faiss_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired session_id. Re-upload documents.",
            )

        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        rag = ConversationalRAG(session_id=session_id) # search and return the user question from the llm
        log.info("Conversational RAG initialized successfully")

        simple_memory = SESSIONS_CHAT_HISTORY.get(session_id, {})
        history = []

        for m in simple_memory:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))

        answer = rag.llm_invoke(message, chat_history=history)
        log.info("RAG invoke success")

        simple_memory.update({"role": "user", "content": message})
        simple_memory.update({"role": "assistant", "content": answer})
        SESSIONS_CHAT_HISTORY[session_id] = simple_memory

        return ChatAnswer(answer=answer)
    except CustomDocumentException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed:{e}")


## Main Function ##

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=2000, reload=True)
