import os # librería para manejar rutas de archivos
import argparse # librería para manejar argumentos de línea de comandos
import joblib # librería para guardar y cargar modelos
import pandas as pd # librería para manejar DataFrames
from datetime import date # librería para manejar fechas

from sklearn.pipeline import Pipeline # librería para crear pipelines de procesamiento

# Para importar funciones de procesamiento de datos
from ai_model.data_processing import (
    load_dataset,
    preprocess_dataframe,
    TextPreprocessor
)

def main():
    parser = argparse.ArgumentParser(
        description="Entrena el pipeline final (individual o ensemble) sobre todo el dataset"
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Ruta al CSV con todos los SMS (e.g. data/eng_sms.csv)'
    )
    parser.add_argument(
        '--model-dir', type=str,
        default=os.path.join(os.path.dirname(__file__), 'saved_models'),
        help='Directorio base donde están los artefactos (saved_models/<idioma>)'
    )
    parser.add_argument(
        '--language', choices=['english', 'spanish'], default='english',
        help='Idioma del dataset y artefactos previos'
    )
    parser.add_argument(
        '--use_lemma', action='store_true',
        help='Usar lematización en el preprocesamiento'
    )
    parser.add_argument(
        '--use_stem', action='store_true',
        help='Usar SnowballStemmer para preprocesamiento en español'
    )
    parser.add_argument(
        '--ensemble', action='store_true',
        help='Reentrena el VotingClassifier en producción'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Umbral de decisión para el modelo (default: 0.5)'
    )
    args = parser.parse_args()

    # 1) Rutas y directorios
    lang_dir = os.path.join(args.model_dir, args.language)
    prod_dir = os.path.join(lang_dir, 'production')
    os.makedirs(prod_dir, exist_ok=True)

    # 2) Carga y filtrado inicial
    print("📥 Cargando y preprocesando dataset completo…")
    df = load_dataset(args.data_path)
    df = preprocess_dataframe(df, use_lemma=args.use_lemma, language=args.language)
    # Normalizar columna label:
    labels = df['label'].astype(str).str.lower().str.strip()
    mapping = {'ham': 0, 'spam': 1, '0': 0, '1': 1}
    def to_int(lbl):
        if lbl in mapping:
            return mapping[lbl]
        else:
            raise ValueError(f"Etiqueta desconocida en y: '{lbl}'")
    df['label'] = labels.map(to_int)


    X = df['message']  # El pipeline interno se encargará de clean + tokenize
    y = df['label']

    today = date.today().strftime('%Y%m%d')

    # 3) Construcción del pipeline
    if args.ensemble:
        # Ensemble: cargamos vectorizador + VotingClassifier
        print("🔄 Cargando artefactos para ensemble…")
        vect_path = os.path.join(lang_dir, 'tfidf_vectorizer.pkl')
        ens_path  = os.path.join(lang_dir, 'ensemble_model.pkl')
        if not os.path.exists(vect_path) or not os.path.exists(ens_path):
            raise FileNotFoundError(
                f"Faltan artefactos de ensemble en {lang_dir}. "
                "Ejecuta primero model_comparison.py con ensemble."
            )
        vect    = joblib.load(vect_path)
        ens_clf = joblib.load(ens_path)
        print(f"✅ Vectorizador cargado: {vect_path}")
        print(f"✅ Ensemble cargado: {ens_path}")

        pipeline = Pipeline([
            ('prep', TextPreprocessor(use_lemma=args.use_lemma, use_stem=args.use_stem)),
            ('vect', vect),
            ('clf',  ens_clf)
        ])
        model_key = 'ensemble'
    else:
        # Individual: cargamos pipeline completo
        best_name_file = os.path.join(lang_dir, 'best_model_name.txt')
        if not os.path.exists(best_name_file):
            raise FileNotFoundError(
                f"No se encontró {best_name_file}. Ejecuta primero model_comparison.py."
            )
        with open(best_name_file, 'r') as f:
            model_key = f.read().strip()
        model_file = f"{model_key}_model.pkl"
        model_path = os.path.join(lang_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el pipeline individual en {model_path}."
            )
        pipeline = joblib.load(model_path)
        print(f"✅ Pipeline individual cargado: {model_path}")

    # 4) Reentrenamiento sobre todo el dataset
    print("🚀 Reentrenando pipeline sobre todo el dataset…")
    pipeline.fit(X, y)

    #4.1) Guardado del pipeline reentrenado
    pipeline.threshold = args.threshold

    os.makedirs(args.model_dir, exist_ok=True)
   # Guardar con sufijo de idioma para no sobrescribir
    fname = f"pipeline_{args.language}.pkl"
    out_path = os.path.join(args.model_dir, fname)
    joblib.dump(pipeline, out_path)
    print(f"✅ Modelo guardado en {out_path} con umbral={pipeline.threshold}")


    # 5) Guardado del pipeline versionado
    out_name = f"{model_key}_prod_{today}.pkl"
    out_path = os.path.join(prod_dir, out_name)
    joblib.dump(pipeline, out_path)
    print(f"✅ Pipeline de producción guardado en: {out_path}")

if __name__ == '__main__':
    main()
