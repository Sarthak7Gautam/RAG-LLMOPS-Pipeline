from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from multi_doc_chat.logging.custom_logger import CustomLogger
from typing import Optional, List
from multi_doc_chat.prompts.prompts import (
    configure_prompt_to_context,
    generate_context_aware_answer,
)
from multi_doc_chat.utils.model_loader import ModelLoader
from langchain_core.documents import Document
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from multi_doc_chat.response_models.models import Validate_AI_Response
import sys
from pydantic import ValidationError
from pathlib import Path
from langchain_community.vectorstores import FAISS

log = CustomLogger().get_custom_logger()


class ConversationalRAG:
    def __init__(
        self, session_id: Optional[str], retriever=None, index_dir="faiss_index"
    ):
        self.session_id = session_id

        self.emb_model = ModelLoader().load_embedding_model()

        self.context_prompt = configure_prompt_to_context
        self.context_answer = generate_context_aware_answer

        self.index_sub_dir = Path(index_dir) / session_id

        self.llm = self._load_llm()

        self.retriever = self.load_retriever_from_FAISS(self.index_sub_dir) or retriever
        self.chain = None
        self.build_lcel_chain()

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()

            if not llm:
                raise ValueError("No LLM found")

            log.info("LLM Model Loaded Successfully")

            return llm

        except Exception as e:
            log.error("Failed to load LLM")
            raise CustomDocumentException(
                "LLM Model Loading Error in Conversational RAG", str(e)
            )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def build_lcel_chain(self):
        try:
            if self.retriever is None:
                log.info("Inside the build_lcel_chain() and no retriever found")
                raise ValueError("No retriever found for building the chain")

            log.info(
                f"The retriever found in build_lcel_chain() and the retriever is {self.retriever}"
            )

            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.context_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = question_rewriter | self.retriever | self.format_docs

            self.chain = (
                (
                    {
                        "context": retrieve_docs,
                        "input": itemgetter("input"),
                        "chat_history": itemgetter("chat_history"),
                    }
                )
                | self.context_answer
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL Graph built successfully")
        except Exception as e:
            raise CustomDocumentException(
                "Failed, Unexpected Error occured in building_lcel_chain", str(e)
            )

    def llm_invoke(
        self, user_input: str, chat_history: Optional[List[BaseMessage]] = None
    ):
        "Invoke the LCEL Pipeline"
        try:
            if self.chain is None:
                raise CustomDocumentException(
                    "RAG Chain Not initialized. Call load_retriever_from_faiss() before llm_invoke()",
                    sys,
                )

            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}

            answer = self.chain.invoke(payload)

            if not answer:
                log.warning("No answer generated")
                return "No answer generated"

            try:
                validated = Validate_AI_Response(res=str(answer))
                answer = validated.res

            except ValidationError as e:
                log.error("Invalid Chat Answer")
                raise CustomDocumentException("Invalid chat Answer", str(e)) from e

            log.info("Chain Invoked Successfully")

            return answer

        except Exception as e:
            log.exception("Failed to llm_invoke Conversational RAG", error=str(e))
            raise CustomDocumentException(
                "Failed to llm_invoke Conversational RAG", sys
            )

    def load_retriever_from_FAISS(self, faiss_sub_dir: Path):
        try:
            vector_store = FAISS.load_local(
                folder_path=str(faiss_sub_dir),
                embeddings=self.emb_model,
                allow_dangerous_deserialization=True,
            )

            if vector_store is None:
                raise CustomDocumentException(
                    "No vector_store found in load_retriever_from_FAISS()"
                )

            log.info("Retriever found in ConversationalRAG")

            self.retriever = vector_store.as_retriever()
            return self.retriever

        except Exception as e:
            log.exception("Failed to load retriever from FAISS", error=str(e))
            raise CustomDocumentException("Failed to load retriever from FAISS", sys)
