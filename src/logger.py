import logging
import os

# --- Papkalarni yaratish ---
os.makedirs('errors', exist_ok=True)
os.makedirs('log', exist_ok=True)

# --- ERROR logger (errors/ papkasiga) ---
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('errors/training_errors.log')
error_handler.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s-%(message)s"))
error_logger.addHandler(error_handler)

# --- INFO logger (log/ papkasiga) ---
info_logger = logging.getLogger('info_logger')
info_logger.setLevel(logging.INFO)
info_handler = logging.FileHandler('log/trainer.log')
info_handler.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s-%(message)s"))
info_logger.addHandler(info_handler)
