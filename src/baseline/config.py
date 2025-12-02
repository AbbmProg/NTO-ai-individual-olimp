from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# -- пути --
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# -- баззовые параметры --
N_SPLITS = 5
RANDOM_STATE = 1487
TARGET = constants.COL_TARGET
# -- разделение бд --

TEMPORAL_SPLIT_RATIO = 0.3995  # лучший результат на проде

# -- обучение --
EARLY_STOPPING_ROUNDS = 80
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"
MODEL_FILENAME = "lgb_model.txt"

# -- tf idf --
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# -- bert  --

BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 64
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.75

# -- признаки по типу id --

CAT_FEATURES = [
constants.COL_USER_ID,
constants.COL_BOOK_ID,
constants.COL_AUTHOR_ID,
constants.COL_PUBLISHER
]

# -- параметры обучения --

LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.85, # 0.80 version 1 & 0.85 version 2
    "bagging_fraction": 0.80, # 0.80 version 1 & 0.80 version 2
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],
}
