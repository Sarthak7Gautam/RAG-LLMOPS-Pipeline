import os
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from multi_doc_chat.logging.custom_logger import CustomLogger
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import Configuration_Loader

log = CustomLogger().get_custom_logger()


class ApiKeyManager:
    def __init__(self):
        load_dotenv()
        self.REQUIRED_KEYS = ["GROQ_API_KEY"]
        self.api_keys = {}

        for key in self.REQUIRED_KEYS:
            val = os.getenv(key)
            self.api_keys[key] = val

        if self.api_keys.items():
            log.info(
                f"API KEY loaded successfully {self.api_keys.get('GROQ_API_KEY')[:12]} "
            )
        else:
            log.error("No API key available in the environment variable")


class ModelLoader:
    _embedding_model = True
    _llm = True

    def __init__(self):
        load_dotenv()
        self.config = Configuration_Loader()
        self.config_data = self.config.load_config()

    def load_embedding_model(self):
        try:
            if self._embedding_model:
                model_provider = self.config_data.get(
                    "embedding_model", {}
                ).get(
                    "provider", {}
                )  # earlier this did not work because the yaml file did not had space after the key, due you which yaml understood it not as a dictionary but as a complete string that said provider:"HuggingFace" and that lead to a malformed config_path and since that config path was a string not a dictionary ans string do not have a .get function and returned None and then I immediately called .get again and python returned the Error None type do not have String and so the code further did not execute not even the log.info for "DEBUG" and hence the code crashed and immediately returned to the exception and raised a Error Loading Embedding Model Exception
                model_name = self.config_data.get("embedding_model", {}).get(
                    "model_name", {}
                )

                log.info(
                    f"DEBUG: The model provider is {model_provider} and the name is {model_name}"
                )

                if model_provider == "HuggingFace":
                    log.info("HuggingFace Embeddings Model Loaded Successfully")
                    return HuggingFaceEmbeddings(model_name=model_name)
                else:
                    log.error("Model Provider not available in the config file")

        except Exception as e:
            raise CustomDocumentException(
                "Error Loading Embeddings Model", str(e)
            ) from None

        self._embedding_model = False

    def load_llm(self):
        try:
            if self._llm:
                log.info(f"The config file is {self.config_data}")

                llm_config = self.config_data.get("llm", {})
                config = llm_config.get("groq", {})
                log.info(f"LLM config is {config}")

                llm_provider = config.get("provider", {})
                llm_model_name = config.get("model_name", {})
                llm_max_output = config.get("max_output_tokens", {})
                llm_temperature = config.get("temperature", {})

                log.debug(
                    f"The llm provider is {llm_provider} and the name is {llm_model_name}"
                )

                self.api_keys = ApiKeyManager()

                if llm_provider == "groq":
                    log.info("ChatGroq Model Initialized Successfully")
                    return ChatGroq(
                        api_key=self.api_keys.api_keys.get("GROQ_API_KEY"),
                        model=llm_model_name,
                        max_tokens=llm_max_output,
                        temperature=llm_temperature,
                    )
                else:
                    log.error("LLM provider not available")

        except Exception as e:
            raise CustomDocumentException(
                "Error initializing ChatGroq Model", str(e)
            ) from None

        self._llm = False
