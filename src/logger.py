# ============================================================
# 1️⃣ BIRINCHI — standart kutubxonalar
import logging
import os

# 2️⃣ IKKINCHI — logging sozlamasi (import lardan keyin darhol)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 3️⃣ KEYIN — boshqa kutubxonalar
import pandas as pd

# 4️⃣ KEYIN — asosiy kod
logger.info("Skript boshlandi")

df = pd.read_csv(r'C:\Users\d1415.csv')
logger.info(f"Fayl o'qildi: {len(df)} qator")