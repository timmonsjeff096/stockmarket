import logging
import logging.handlers
import os

import data_processing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "C:\Python\Projects\stockmarket\logs\stocklog.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
ch.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(ch)
