# data_processing.py

"""
Módulo unificado para procesamiento de SMS:
 - Funciones clásicas de carga, limpieza y preprocesado para TF-IDF + sklearn.
 - Clases y utilidades para fine-tuning e inferencia con BERT.
 - Integración de Google Safe Browsing API para tokenización de URLs.
"""

import os
import re
import unidecode
import requests
import pandas as pd
import numpy as np
import torch
from functools import lru_cache
from typing import List
from sklearn.base import TransformerMixin, BaseEstimator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.config import settings

# ------------------------------
# --- Configuración de modelos BERT por idioma (fallback a HF) ---
SPANISH_MODEL = os.getenv("SPANISH_BERT_MODEL", "dccuchile/bert-base-spanish-wwm-cased")
ENGLISH_MODEL = os.getenv("ENGLISH_BERT_MODEL", "bert-base-uncased")

# Mapa directo de argumento --language al código y carpeta local
DIR_MAP = {
    "spanish": ("es", "bert_sms_detector_es"),
    "english": ("en", "bert_sms_detector_en"),
}
# ------------------------------
# Configuración Safe Browsing
# ------------------------------
API_KEY = settings.GOOGLE_SAFE_BROWSING_API_KEY
SAFE_BROWSING_URL = (
    "https://safebrowsing.googleapis.com/v4/threatMatches:find"
)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
@lru_cache(maxsize=10000)
def check_url_reputation(url: str) -> bool:
    body = {
        "client": {"clientId": "DetectorSMS", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    params = {"key": API_KEY}
    try:
        resp = requests.post(SAFE_BROWSING_URL, json=body, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return "matches" in data and bool(data["matches"])
    except requests.RequestException:
        return False

def extract_urls(text: str) -> list[str]:
    """Devuelve la lista de URLs encontradas en el texto."""
    return URL_REGEX.findall(text)

def replace_url_with_token(text: str) -> str:
    def _repl(match):
        url = match.group(0)
        return ' URL_SUSPICIOUS ' if check_url_reputation(url) else ' URL_TRUSTED '
    return URL_REGEX.sub(_repl, text)

# ------------------------------
# Funciones de limpieza
# ------------------------------

def clean_text(text: str) -> str:
    # Tokeniza URLs con reputación
    text = replace_url_with_token(text)
    # Eliminar acentos
    text = unidecode.unidecode(text)
    # Reemplazar números largos
    text = re.sub(r"\b\d{10,}\b", ' PHONE_NUM ', text)
    # Eliminar puntuación
    text = re.sub(r"[^\w\s]", ' ', text)
    # Minusculas y espacios
    return text.lower().strip()

# ------------------------------
# Dataset clásico
# ------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga CSV con columnas 'label' y 'message' o viceversa.
    Normaliza nombres a ['label','message'] y retorna DataFrame.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    # Asegurar columnas 'label' y 'message'
    if 'label' in cols and 'message' in cols:
        df = df.rename(columns={c: c for c in cols})
    else:
        # si no, reasignar primeras dos columnas
        df.columns = ['label', 'message']
    # Seleccionar y filtrar nulos
    df = df[['message', 'label']]
    df = df.dropna(subset=['message', 'label'])
    return df


def map_label(label) -> int:
    """
    Mapea etiquetas mixtas: 'spam','ham','0','1', int
    """
    if isinstance(label, str):
        l = label.lower().strip()
        if l in ('spam', '1'):
            return 1
        if l in ('ham', '0'):
            return 0
    elif isinstance(label, (int, np.integer)):
        return int(label)
    raise ValueError(f"Etiqueta desconocida: {label}")


def preprocess_dataframe(
    df: pd.DataFrame,
    use_lemma: bool = False,
    use_stem: bool = False,
    language: str = 'english'
) -> pd.DataFrame:
    """
    Aplica map_label y clean_text.
    Retorna DataFrame con columnas ['message','label'] limpias.
    """
    df = df.copy()
    # Mapear etiquetas
    df['label'] = df['label'].apply(map_label)
    # Limpiar texto
    df['message'] = df['message'].apply(clean_text)
    # Quitar posibles filas vacías
    df = df[df['message'].str.strip().astype(bool)]
    return df.reset_index(drop=True)

# ------------------------------
# Preprocesador para Pipeline sklearn
# ------------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Limpia texto y opcionalmente aplica stemming o lematización.
    Usa clean_text internamente.
    """
    def __init__(
        self,
        use_lemma: bool = False,
        use_stem: bool = False,
        language: str = 'english'
    ):
        self.use_lemma = use_lemma
        self.use_stem = use_stem
        # Importar librerías NLTK según necesidad
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        self.stemmer = PorterStemmer() if use_stem else None
        self.lemmatizer = WordNetLemmatizer() if use_lemma else None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        results = []
        for text in X:
            txt = clean_text(text)
            tokens = txt.split()
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens]
            if self.stemmer:
                tokens = [self.stemmer.stem(tok) for tok in tokens]
            results.append(' '.join(tokens))
        return results

# ------------------------------
# Funciones BERT
# ------------------------------
@lru_cache(maxsize=2)
def load_bert_model(language: str):
    """
    language: 'spanish' o 'english'
    Primero busca carpeta local bert_sms_detector_{es,en}.
    Si existe, carga de ahí. Si no, descarga el modelo HF por defecto.
    """
    key = language.lower()
    if key not in DIR_MAP:
        raise ValueError(f"Idioma desconocido para BERT: {language!r}. Usa 'spanish' o 'english'.")
    code, local_dirname = DIR_MAP[key]
    # 1) Intentar carpeta local
    local_path = os.path.join(os.path.dirname(__file__), local_dirname)
    if os.path.isdir(local_path):
        return (
            AutoTokenizer.from_pretrained(local_path),
            AutoModelForSequenceClassification.from_pretrained(local_path)
        )
    # 2) Fallback: modelo HF
    hf_model = SPANISH_MODEL if code == "es" else ENGLISH_MODEL
    return (
        AutoTokenizer.from_pretrained(hf_model),
        AutoModelForSequenceClassification.from_pretrained(hf_model)
    )

def predict_spam(texts: List[str], language: str = "english", batch_size: int = 64) -> List[float]:
    tokenizer, model = load_bert_model(language)
    model.eval()
    probs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch,
                          padding=True,
                          truncation=True,
                          max_length=128,
                          return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        probs.extend(batch_probs)
        # feedback de progreso
        print(f"  → Procesados {min(i+batch_size, total)}/{total} mensajes")
    return probs





