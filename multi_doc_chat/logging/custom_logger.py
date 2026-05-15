import structlog
from pathlib import Path
import logging
from datetime import datetime

"""
The exception block (the big part with the line numbers and file names) only appears when you use logger.exception().For info and error, structlog sees that there is no active crash to report, so it simply skips the dict_tracebacks part and gives you a nice, clean, short JSON object. This keeps your log file from getting bloated with useless info for regular events.
"""


class CustomLogger:
    _initialized = False

    def __init__(self, logs_dir="logs"):

        if CustomLogger._initialized:
            return

        self.log_base = Path(logs_dir)
        self.log_base.mkdir(exist_ok=True, parents=True)
        log_file = (
            self.log_base / f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
        )

        logging.basicConfig(  # this carries the logging configuration you just created to the python standard logging
            level=logging.INFO,
            format="%(message)s",  # here you tell the python logging to not add any info from itself just take what I have given you
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],  # since we are using handlers logging now cannot log automatically to the screen so we have to provide streamhandler to show logs on the screen
        )

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.stdlib.add_log_level,
                structlog.processors.dict_tracebacks ,
                structlog.processors.EventRenamer(to="msg"),
                structlog.processors.JSONRenderer(indent = 4),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),  # It connects structlog to python logging to get JSON output
            wrapper_class=structlog.stdlib.BoundLogger,  # gives the log a memory like if you use bind(id=20) this id will appear every time you wont have to type it again and again
            cache_logger_on_first_use=True,
        )

        CustomLogger._initialized = True

    def get_custom_logger(
        self,
    ):  # this gives you the finished pen that gives you full control on how to write logs without having to retype the original code again like logger = get_logger()
        return structlog.get_logger()
