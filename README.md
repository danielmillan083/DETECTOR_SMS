# 📱 Detector de Fraude en SMS con IA

Sistema completo para la detección automática de SMS fraudulentos utilizando modelos de NLP clásicos y avanzados (Random Forest, BERT), con backend en FastAPI, interfaz en Streamlit y app móvil en Flutter.

---

## 🧠 Arquitectura del Proyecto

/ai-model/         # Entrenamiento, evaluación y preprocesamiento
/backend/          # API REST (FastAPI) y gestión de base de datos
/mobile-app/       # App Flutter (detección local o vía API)
/docs/             # Documentación técnica y diseño

---

## 🚀 Características Principales

- Modelos clásicos (`RandomForest`, `NaiveBayes`, `SVM`, etc.) y `BERT` fine-tuneado.
- Clasificación de mensajes como `legítimo`, `posible_spam`, o `fraude`.
- Evaluación automática: `accuracy`, `precision`, `recall`, `F1`, `ROC`.
- Backend FastAPI con endpoints `/predict`, `/feedback`, `/block_number`.
- App Flutter con análisis embebido (TFLite) o vía API (no desarrolada todavía).
- Interfaz adicional con Streamlit para pruebas locales.
- Reputación de URLs usando Google Safe Browsing API.
- Feedback de usuario para reentrenamiento futuro.

---

## 📦 Requisitos

### Backend

Python >= 3.8
pip install -r requirements.txt

### App móvil (Flutter)

Flutter SDK >= 3.x

---

## ⚙️ Entrenamiento de Modelos

### Ensemble voting (recopilación de los cuatro modelos Random Forest, Logistic Regression, Naive Bayes y SVM en uno solo)

#### Para ejecutar con idioma en inglés

python -m ai_model.model_comparison --data_path data/eng_sms_train.csv --language english --use_lemma
python -m ai_model.train_production --data_path data/eng_sms_train.csv --language english --use_lemma --ensemble

#### Mejor modelo guardado (SVM)

python -m ai_model.train_production --data_path data/eng_sms_train.csv --language english --use_lemma

#### Para ejecutar con idioma en español

python -m ai_model.model_comparison --data_path data/esp_sms_clean_train.csv --language spanish --use_stem
python -m ai_model.train_production --data_path data/esp_sms_clean_train.csv --language spanish --use_stem --ensemble --threshold 0.429

#### Mejor modelo guardado  español (SVM)

python -m ai_model.train_production --data_path data/esp_sms_clean_train.csv --language spanish --use_stem --threshold 0.429

### BERT (fine-tuning)

#### Inglés

python -m ai_model.train_production_bert --lang english --train-file data/eng_sms_train.csv --output-dir ai_model/bert_sms_detector_en
python -m ai_model.model_comparison --data_path data/eng_sms_train.csv --language english --with_bert

#### Español

python -m ai_model.train_production_bert --lang spanish --train-file data/esp_sms_clean_train.csv --output-dir ai_model/bert_sms_detector_es
python -m ai_model.model_comparison --data_path data/esp_sms_clean_train.csv --language spanish --with_bert

## 🧪 Evaluación

### Ensemble voting (inglés)

python -m ai_model.run_evaluation --data_path data/eng_sms_test.csv --language english --use_lemma --ensemble

### Mejor modelo (SVM) (inglés)

python -m ai_model.run_evaluation --data_path data/eng_sms_test.csv --language english --use_lemma

### BERT (inglés)

python -m ai_model.run_evaluation --data_path data/eng_sms_test.csv --language english --with_bert

### Ensemble voting (español)

python -m ai_model.run_evaluation --data_path data/esp_sms_clean_test.csv --language spanish --use_stem --ensemble

### Mejor modelo (SVM) (español)

python -m ai_model.run_evaluation --data_path data/esp_sms_clean_test.csv --language spanish --use_stem

### BERT (español)

python -m ai_model.run_evaluation --data_path data/esp_sms_clean_test.csv --language spanish --with_bert

## 🌐 API REST (FastAPI)

### Ejecutar localmente

uvicorn app:app --reload

### Endpoints útiles

- `POST /predict` → analiza un SMS
- `POST /feedback` → guarda retroalimentación
- `POST /block_number` → bloquea un número
- `GET /blocked_numbers` → lista de bloqueados
- `GET /health` → chequeo de salud

---

## 🖥️ Interfaz Streamlit

streamlit run detectorsms_streamlit.py

---

## 📱 App Flutter (no desarrollada todavía)

En `mobile-app/lib/` se encuentran los archivos principales.

### Permisos (Android)

```xml
<uses-permission android:name="android.permission.READ_SMS" />
<uses-permission android:name="android.permission.RECEIVE_SMS" />
```

### Funcionalidades

- Lectura de SMS en tiempo real
- Llamada a API REST o uso de modelo `.tflite`
- Visualización de riesgo con colores (semaforización)
- Envío de feedback

---

## 🛢️ Base de Datos

Usa SQLAlchemy + SQLite (modo desarrollo)

Tablas:

- `sms_messages`: historial de análisis
- `blocked_numbers`: números bloqueados
- `feedback`: respuestas de usuarios

---

## 📂 Variables de entorno (`.env`)

```env
DATABASE_URL=sqlite:///./sms.db
GOOGLE_SAFE_BROWSING_API_KEY=TU_API_KEY
```

---

## 📈 Ejemplo de Uso

curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Has ganado un premio, reclama aquí https://bit/ly.com", "language": "spanish"}'

---

## 🧑‍💻 Créditos

Proyecto desarrollado como solución integral de detección de fraude en SMS con IA.  
Incluye frontend, backend, entrenamiento de modelos, y base de datos integrada.

---

## 📜 Licencia

MIT License
