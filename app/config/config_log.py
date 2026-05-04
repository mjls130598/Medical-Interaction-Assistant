import logging
import sys

def setup_logging():
    # Evita que se configure más de una vez
    if logging.getLogger().hasHandlers():
        return

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout), # Consola
            logging.FileHandler("app.log", encoding="utf-8") # Archivo
        ]
    )