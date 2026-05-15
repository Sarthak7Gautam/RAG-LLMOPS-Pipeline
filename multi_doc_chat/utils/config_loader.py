from pathlib import Path
import yaml
from multi_doc_chat.logging.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException

log = CustomLogger().get_custom_logger()


class Configuration_Loader:
    def project_root(self):
        return Path(__file__).resolve().parents[1]

    def load_config(self):
        "this code has a config_path logic which is used for docker in production level applications I am not implementing them right now I am just writing the main logic for now but later while doing CI/CD I will implement it"

        config_file_path = self.project_root() / "config" / "config.yaml"

        try:
            with open(config_file_path, "r") as f:
                log.info("Config File loaded successfully")
                return yaml.safe_load(f)

        except FileNotFoundError as e:
            raise CustomDocumentException("File not found", str(e)) from None
