import os
import joblib
import torch
from tensorflow.keras.saving import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings

# ===============================
# BASE DIRECTORY (CRITICAL)
# ===============================
BASE_DIR = settings.BASE_DIR

# ===============================
# PATHS (ABSOLUTE, NOT RELATIVE)
# ===============================
BILSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "bilstm_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "bilstm_tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "bilstm_label_encoder.pkl")
BERT_PATH = os.path.join(BASE_DIR, "models", "bangla_bert_model")

# ===============================
# LOAD Bi-LSTM
# ===============================
BILSTM_MODEL = load_model(BILSTM_MODEL_PATH, compile=False)
BILSTM_TOKENIZER = joblib.load(TOKENIZER_PATH)
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)

# ===============================
# LOAD BERT
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_PATH)
BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
BERT_MODEL.to(DEVICE)
BERT_MODEL.eval()
