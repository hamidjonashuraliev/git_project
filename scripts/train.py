import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import error_logger, info_logger


# scripts/train.py ichiga yozing
from src.logger import error_logger, info_logger

info_logger.info("Model trening boshlandi")
error_logger.error("Xatolik yuz berdi!")