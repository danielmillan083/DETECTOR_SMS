#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evaluation.py

Carga y evalúa el pipeline final —ya sea el mejor modelo individual,
el ensemble de sklearn o el BERT fine-tuneado— para detección de fraude en SMS.
Genera métricas, matriz de confusión y curva ROC.
"""

import os
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from ai_model.data_processing import (
    load_dataset,
    preprocess_dataframe,
    TextPreprocessor,
    predict_spam
)
from ai_model.evaluation_utils import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve
)

def main():
    parser = argparse.ArgumentParser(
        description="Evalúa pipeline SMS fraud detector (sklearn o BERT)"
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Ruta al CSV de SMS de test (e.g. data/esp_sms_test.csv)'
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=os.path.join(os.path.dirname(__file__), 'saved_models'),
        help='Directorio base con artefactos en saved_models/<idioma>'
    )
    parser.add_argument(
        '--language', choices=['english', 'spanish'], default='english',
        help='Idioma para preprocesamiento y stopwords'
    )
    parser.add_argument(
        '--use_lemma', action='store_true',
        help='Usar lematización en el TextPreprocessor (solo afecta a sklearn)'
    )
    parser.add_argument(
        '--use_stem', action='store_true',
        help='Usar stemming en el TextPreprocessor (solo en español)'
    )
    parser.add_argument(
        '--ensemble', action='store_true',
        help='Para pipelines sklearn: evaluar el VotingClassifier en lugar del mejor individual'
    )
    parser.add_argument(
        '--with_bert', action='store_true',
        help='Evaluar BERT fine-tuneado en lugar de pipelines sklearn'
    )
    args = parser.parse_args()

    # 1) Carga de datos
    print(f"\n📥 Cargando datos de: {args.data_path}")
    df = load_dataset(args.data_path)  # limpieza de filas vacías :contentReference[oaicite:0]{index=0}
    df = preprocess_dataframe(df,
                              use_lemma=args.use_lemma,
                              language=args.language)
    # Normalizar etiquetas mixtas ('ham','spam','0','1')
    df['label'] = (df['label']
                   .astype(str)
                   .str.strip()
                   .str.lower()
                   .map({'ham': 0, 'spam': 1, '0': 0, '1': 1}))

    X = df['message']
    y = df['label']

    # 2) Si pedimos evaluar BERT, saltamos sklearn
    if args.with_bert:
        print("\n===== Evaluación BERT fine-tuneado =====")
        # predict_spam recibe lista de strings y devuelve lista de probabilidades de spam
        probs = predict_spam(X.tolist(), language=args.language)
        preds = [int(p >= 0.5) for p in probs]
        # Métricas + gráficas
        evaluate_model(y, preds, probs)
        plot_confusion_matrix(y, preds, labels=["Ham", "Spam"])
        plot_roc_curve(y, probs)
        print("\n✔️ Evaluación BERT completada.")
        return

    # 3) Para sklearn, reconstrucción del pipeline
    model_dir = os.path.join(args.model_dir, args.language)
    print(f"\n🔍 Buscando artefactos sklearn en: {model_dir}")

    if args.ensemble:
        # Ensemble de VotingClassifier + vectorizador
        vect_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        ens_path  = os.path.join(model_dir, 'ensemble_model.pkl')
        if not os.path.exists(vect_path) or not os.path.exists(ens_path):
            raise FileNotFoundError(
                f"Faltan artefactos de ensemble en {model_dir}. "
                "Ejecuta primero model_comparison.py con --with-bert=False."
            )
        vect    = joblib.load(vect_path)
        ens_clf = joblib.load(ens_path)
        pipeline = Pipeline([
            ('prep', TextPreprocessor(use_lemma=args.use_lemma,
                                      use_stem=args.use_stem,
                                      language=args.language)),
            ('vect', vect),
            ('clf',  ens_clf)
        ])
        print("✅ Pipeline ensemble reconstruido.")
    else:
        # Mejor modelo individual
        best_name_file = os.path.join(model_dir, 'best_model_name.txt')
        if not os.path.exists(best_name_file):
            raise FileNotFoundError(
                f"No se encontró best_model_name.txt en {model_dir}. "
                "Ejecuta primero model_comparison.py."
            )
        with open(best_name_file, 'r') as f:
            model_key = f.read().strip()
        model_file = f"{model_key}_model.pkl"
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el pipeline individual en {model_path}."
            )
        pipeline = joblib.load(model_path)
        print(f"✅ Pipeline individual cargado: {model_file}")

    # 4) Predicción y evaluación (sklearn)
    print("\n🚀 Realizando predicciones con sklearn…")
    y_pred = pipeline.predict(X)
    y_prob = (pipeline.predict_proba(X)[:, 1]
              if hasattr(pipeline, 'predict_proba') else None)

    evaluate_model(y, y_pred, y_prob)
    plot_confusion_matrix(y, y_pred, labels=["Ham", "Spam"])
    plot_roc_curve(y, y_prob)

    print("\n✔️ Evaluación sklearn completada.")

if __name__ == '__main__':
    main()


