# data ingestion involes steps like doc loaders, cleaning, text_split, embedding_generation, store in vector database and this file does this

import uuid
from datetime import datetime
from pathlib import Path
from multi_doc_chat.logging.custom_logger import CustomLogger
from multi_doc_chat.utils.model_loader import ModelLoader
from typing import Optional, Dict, List, Iterable
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from multi_doc_chat.utils.uploaded_file_saving import save_uploaded_files
from multi_doc_chat.utils.document_loader import Load_Documents
import json
from langchain_community.vectorstores import FAISS
import hashlib
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

log = CustomLogger().get_custom_logger()


def generate_session_id() -> str:
    "Generate a TimeStamp based Unique Session Id"
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = uuid.uuid4().hex[:4]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    def __init__(
        self,
        user_uploaded_data_temp_base: str = "data",
        faiss_vector_store_base: str = "faiss_index",
        use_sessions_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader().load_embedding_model()

            self.use_sessions_dirs = use_sessions_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(user_uploaded_data_temp_base)
            self.temp_base.mkdir(exist_ok=True, parents=True)

            self.faiss_base = Path(faiss_vector_store_base)
            self.faiss_base.mkdir(exist_ok=True, parents=True)

            self.temp_sub_dir = self._resolve_sub_dir(self.temp_base)
            self.faiss_sub_dir = self._resolve_sub_dir(self.faiss_base)

            log.info("Chat Ingestor Initialized Successfully")
        except Exception as e:
            log.error("Chat Ingestor Not Initialized, Error Occured")

            raise CustomDocumentException(
                "Error occured while initializing ChatIngestor", str(e)
            ) from e

    def _resolve_sub_dir(self, base: Path) -> Path:
        if self.use_sessions_dirs:
            dir = base / self.session_id
            dir.mkdir(exist_ok=True, parents=True)
            return dir
        return base

    def _split(
        self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 250
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents=docs)
        log.info("Documents split complete and chunks created successfully")
        return chunks

    def built_retriever(
        self,
        uploaded_files: Iterable,
        chunk_size=1000,
        chunk_overlap=250,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 15,
        lambda_mult: float = 0.5,
    ):
        try:
            path = save_uploaded_files(
                uploaded_files=uploaded_files, target_dir=self.temp_sub_dir
            )

            loader = Load_Documents()

            docs = loader.load_docs(path)

            if not docs:
                raise ValueError("No valid documents found")

            chunks = self._split(
                docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            fm = FaissManager(
                self.faiss_sub_dir, self.model_loader
            )  # stores_meta_data, creates_vector_store, add_docs_in_vector_store
            log.info("Faiss Manager Initialized")

            self.vs = None
            try:
                self.vs = fm.load_or_create_vector_store()
                log.info(
                    "Vector Store load_or_create_vector_store() successfull in the ChatIngestor"
                )

            except Exception as e:
                raise CustomDocumentException(
                    "Exception occured while load_or_create_vector_store()", str(e)
                )

            added = fm.add_documents_in_vector_store(docs=chunks, vector_store=self.vs)

            log.info("FAISS index updated successfully", added = added)

            if search_type == "mmr":
                search_kwargs = {"k": k}
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR for similarity search")

            retriever = self.vs.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

            log.info(
                "Retriever built successful, Documents retrieved successfully for retrieval"
            )
            return retriever

        except Exception as e:
            raise CustomDocumentException(
                "Error occured while building retriever", str(e)
            ) from e


class FaissManager:
    def __init__(
        self, index_dir: Path, model_loader: Optional[ModelLoader] = None
    ):  # index_dir == faiss_sub_dir
        # these below two lines are not needed but it is safe
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True, parents=True)

        self.meta_path = index_dir / "ingested_meta.json"
        self._meta: Dict[str, any] = {"data": {}}

        if self.meta_path.exists():
            self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {
                "data": {}
            }
        else:
            self._meta = {"data": {}}

        log.info("Meta data stored successfully in the faiss sub directory")
        self.emb_model = model_loader
        self.vector_store: Optional[FAISS] = None

    def load_or_create_vector_store(self):
        "Loads vector_store if it exists for that particular document else creates new vector_store"
        try:
            if self._index_exists():
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.emb_model,
                    allow_dangerous_deserialization=True,
                )

                return self.vector_store
            else:
                # Initialize with one temporary doc
                test_embed = self.emb_model.embed_query("test")
                dimension = len(test_embed)

                # 2. Create a truly empty FAISS index
                index = faiss.IndexFlatL2(dimension)

                self.vector_store = FAISS(
                    embedding_function=self.emb_model,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                return self.vector_store

        except Exception as e:
            raise CustomDocumentException(
                "Error in the load_or_create_vector_store()", str(e)
            )

    def add_documents_in_vector_store(
        self, docs: list[Document], vector_store: FAISS = None
    ):
        "Returns number of new added docs in vector_store and saves docs in vector_store"
        self.vector_store = vector_store

        if self.vector_store is None:
            raise CustomDocumentException(
                "Call load_or_create_vector_store() to create vector_store before adding documents"
            )

        new_docs: List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key not in self._meta["data"]:
                self._meta["data"][key] = True
                new_docs.append(d)

        if new_docs:
            log.info(f"DEBUG: The new documents are {new_docs[0].page_content}")
            self.vector_store.add_documents(new_docs)
            self.vector_store.save_local(
                str(self.index_dir)
            )  # this line creates faiss.index and faiss.pkl
            self._save_meta()  # in ingested_meta.json

        log.info(f"No of updated docs: {len(new_docs)}")
        return len(new_docs)

    def _index_exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "index.pkl"
        ).exists()

    @staticmethod
    def _fingerprint(text: str, metadata: Dict[str, any]) -> str:
        src = metadata.get("source") or metadata.get("file_path")
        row_id = metadata.get("row_id")
        if src is not None:
            return f"{src}::{' ' if row_id is None else row_id}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        log.info("Meta data saved successfully inside save_meta")
        return self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
