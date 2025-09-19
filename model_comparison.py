#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_comparison.py

Compara clasificadores clásicos con TF-IDF y un posible modelo BERT fine-tuneado
para detección de fraude en SMS, tanto en español como en inglés.
Guarda el mejor modelo individual y un ensemble de soft voting con sus vectores.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from ai_model.data_processing import (
    load_dataset,
    preprocess_dataframe,
    TextPreprocessor,
    predict_spam
)

# Configuración de modelos clásicos y sus parámetros
MODELS_CONFIG = {
    'random_forest': {
        'estimator': RandomForestClassifier,
        'params': {'n_estimators': 100, 'random_state': 42}
    },
    'logistic_regression': {
        'estimator': LogisticRegression,
        'params': {'max_iter': 1000}
    },
    'naive_bayes': {
        'estimator': MultinomialNB,
        'params': {}
    },
    'svm': {
        'estimator': SVC,
        'params': {'kernel': 'linear', 'probability': True}
    }
}


def make_text_pipeline(max_features, use_lemma, use_stem, language, estimator):
    """
    Pipeline de preprocesado y clasificación sin lambdas:
      - TextPreprocessor
      - TfidfVectorizer
      - clasificador
    """
    return Pipeline([
        ('prep', TextPreprocessor(use_lemma=use_lemma,
                                  use_stem=use_stem,
                                  language=language)),
        ('vect', TfidfVectorizer(max_features=max_features)),
        ('clf', estimator)
    ])


def print_metrics(model_name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n===== {model_name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC      : {auc:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    parser = argparse.ArgumentParser(
        description="Compara modelos clásicos y opcionalmente BERT"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="CSV de entrenamiento"
    )
    parser.add_argument(
        "--use_lemma", action="store_true",
        help="Usar lematización en TextPreprocessor"
    )
    parser.add_argument(
        "--use_stem", action="store_true",
        help="Usar stemming en TextPreprocessor (solo español)"
    )
    parser.add_argument(
        "--language", choices=["english", "spanish"],
        default="english",
        help="Idioma para preprocesado"
    )
    parser.add_argument(
        "--with_bert", action="store_true",
        help="Incluir evaluación de modelo BERT"
    )
    args = parser.parse_args()

    # Carga y preprocesado dataframe
    df = load_dataset(args.data_path)
    df = preprocess_dataframe(
        df,
        use_lemma=args.use_lemma,
        use_stem=args.use_stem,
        language=args.language
    )

    # División train / test
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Directorio de guardado
    model_dir = os.path.join(
        os.path.dirname(__file__), 'saved_models', args.language
    )
    os.makedirs(model_dir, exist_ok=True)

    # Comparación de modelos clásicos
    best_acc = -1.0
    best_pipeline = None
    for name, cfg in MODELS_CONFIG.items():
        estimator_cls = cfg['estimator']
        params = cfg['params']
        # Instancia limpia para pipeline
        estimator = estimator_cls(**params)
        pipe = make_text_pipeline(
            max_features=10000,
            use_lemma=args.use_lemma,
            use_stem=args.use_stem,
            language=args.language,
            estimator=estimator
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        print_metrics(name, y_test, y_pred, y_prob)

        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_pipeline = (name, pipe)

    # Guardar mejor modelo individual
    if best_pipeline:
        name, pipeline = best_pipeline
        fname = f"{name}_model.pkl"
        joblib.dump(pipeline, os.path.join(model_dir, fname))
        with open(os.path.join(model_dir, "best_model_name.txt"), 'w') as f:
            f.write(name)
        print(f"\n🏆 Mejor modelo ({name}) guardado en {fname}")

    # Ensemble soft voting usando vect y prep del mejor pipeline
    name, pipeline = best_pipeline
    prep = pipeline.named_steps['prep']
    vect = pipeline.named_steps['vect']
    X_tr_proc = prep.transform(X_train)
    X_te_proc = prep.transform(X_test)
    X_tr_vec = vect.fit_transform(X_tr_proc)
    X_te_vec = vect.transform(X_te_proc)

    ensemble = VotingClassifier(
        estimators=[(n, cfg['estimator'](**cfg['params']))
                    for n, cfg in MODELS_CONFIG.items()],
        voting='soft'
    )
    ensemble.fit(X_tr_vec, y_train)
    y_pred_ens = ensemble.predict(X_te_vec)
    y_prob_ens = ensemble.predict_proba(X_te_vec)[:, 1]
    print_metrics("Ensemble", y_test, y_pred_ens, y_prob_ens)

    # Guardar vectorizador y ensemble
    joblib.dump(vect, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(ensemble, os.path.join(model_dir, "ensemble_model.pkl"))
    print("✅ Ensemble y vectorizador guardados.")

    # Evaluación de BERT
    if args.with_bert:
        probs_bert = predict_spam(X_test.tolist(), language=args.language)
        preds_bert = [int(p >= 0.5) for p in probs_bert]
        print_metrics("BERT", y_test, preds_bert, probs_bert)


if __name__ == "__main__":
    main()



