# test_data_processing.py

import os
from ai_model.data_processing import predict_spam, clean_text, check_url_reputation
from dotenv import load_dotenv
# 1) Asegúrate de que la API key está en la variable de entorno
#    export GOOGLE_SAFE_BROWSING_API_KEY="tu_api_key"
#assert "GOOGLE_SAFE_BROWSING_API_KEY" in os.environ, "Falta la API key"

load_dotenv()  # Cargar variables de entorno desde .env
# 2) Cargar la API key
API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
if not API_KEY:
    raise ValueError("Falta la API key de Google Safe Browsing. Asegúrate de que está en .env")

# 2) Mensajes de ejemplo
ejemplos_es = [
    "Estimado cliente, su pedido no llegó. Consulte aquí: http://mistracking.es/123",
    "Hola, ¿qué tal? Este es un mensaje de prueba sin spam."
]
ejemplos_en = [
    "Dear customer, your package couldn't be delivered. Click http://phish.link",
    "Hey, just checking in with you!"
]

# 3) Limpieza y reputación de URLs
print("=== clean_text + URL reputation ===")
for txt in ejemplos_es + ejemplos_en:
    limpio = clean_text(txt)
    print(f">>> ORIGINAL: {txt!r}")
    print(f"    CLEANED: {limpio!r}\n")

# 4) Probabilidades de spam
print("=== predict_spam (ES) ===", predict_spam(ejemplos_es, lang="es"))
print("=== predict_spam (EN) ===", predict_spam(ejemplos_en, lang="en"))
