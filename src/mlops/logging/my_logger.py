from loguru import logger
import sys
logger.remove()
logger.add(sink="loguru/loguru_logs.log", rotation="1 MB", level='WARNING')
for i in range(10000):
    logger.debug(f"{i} - Loguru logger")
    logger.warning(f"{i} - Loguru logger")
    logger.error(f"{i} - Loguru logger")
    logger.critical(f"{i} - Loguru logger")
    logger.info(f"{i} - Loguru logger")
