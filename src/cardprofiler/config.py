from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_card_data.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "df_clean.csv"
MODEL_PATH = MODELS_DIR / "best_pipeline.joblib"

# Machine learning
RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE_THRESHOLD = 0.80
