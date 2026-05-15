from multi_doc_chat.exceptions.custom_exceptions import CustomDocumentException
from typing import Iterable
from pathlib import Path
from multi_doc_chat.logging.custom_logger import CustomLogger
import yaml

log = CustomLogger().get_custom_logger()


def project_root():
    return Path(__file__).resolve().parents[1]


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path):

    try:
        "Saves the uploaded files in the target directory"

        target_dir.mkdir(parents=True, exist_ok=True)
        config_path = project_root() / "config" / "config.yaml"

        with open(config_path, "r") as a:
            file = yaml.safe_load(a)  # this file holds a dictionary

        SUPPORTED_EXTENSIONS = file["SupportedExtensions"]
        file_save_path = None 

        for uf in uploaded_files: # uf is a FastApiFileAdapter
            filename = getattr(
                uf, "filename", getattr(uf, "file", getattr(uf, "name", "unknown_file"))
            )  # error was "name" should have been before because "unknown_file" will not be an attribute it will be a default string which will not have an ext so error will be thrown, if you want to check another attribute as well add another getattr

            log.info(f"The filename is {filename}")
            ext = Path(filename).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log.info("File not supported")
                continue

            try:
                content = uf.get_buffer()

                file_save_path = target_dir / filename
                file_save_path.write_bytes(content)

                log.info(f"Successfully saved {filename}")
            except Exception as e:
                log.error(f"Failed to save file: {e}")

        log.info(f"The user uploaded file saved successfully in {file_save_path}")
        return file_save_path
    except Exception as e:
        raise CustomDocumentException("Error occured while saving the uploaded file", e) from e 