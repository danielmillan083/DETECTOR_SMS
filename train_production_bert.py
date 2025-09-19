# train_production_bert.py

"""
Script para el fine-tuning de un modelo BERT para detección de fraude en SMS.
Soporta español e inglés, y guarda el modelo y tokenizer entrenados.
Incluye mapeo robusto de etiquetas mixtas ('ham','spam',0,1).
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from ai_model.data_processing import clean_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de BERT para detección de fraude en SMS"
    )
    parser.add_argument(
        "--lang", choices=["spanish", "english"], required=True,
        help="Idioma del dataset: 'spanish' o 'english'"
    )
    parser.add_argument(
        "--train-file", type=str, required=True,
        help="Ruta al CSV de entrenamiento con columnas 'message' y 'label' (spam/ham o 0/1)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directorio donde guardar el modelo fine-tuneado"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Número de épocas de entrenamiento"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Tamaño de batch para entrenamiento y evaluación"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate para el optimizador"
    )
    parser.add_argument(
        "--split", type=float, default=0.1,
        help="Proporción de validación (0.0-1.0)"
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }


def map_label(lab):
    """
    Convierte etiquetas mixtas en enteros {0,1}:
    - 'spam' (str) o 1 (int/str) -> 1
    - 'ham' (str)  or 0 (int/str) -> 0
    """
    # Si es número
    if isinstance(lab, (int, float)):
        val = int(lab)
        if val in (0, 1):
            return val
    # Si es cadena
    if isinstance(lab, str):
        lab_str = lab.lower().strip()
        if lab_str == 'spam' or lab_str == '1':
            return 1
        if lab_str == 'ham' or lab_str == '0':
            return 0
    raise ValueError(f"Etiqueta inesperada: {lab}")


def main():
    args = parse_args()

    # 1) Carga de datos
    df = pd.read_csv(args.train_file)
    # Aseguramos que existan las columnas
    if 'message' not in df.columns or 'label' not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'message' y 'label'.")

    # 2) Convertir a Dataset y split
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=args.split, seed=42)
    ds_train = ds['train']
    ds_val   = ds['test']

    # 3) Selección de modelo
    model_name = 'dccuchile/bert-base-spanish-wwm-cased' if args.lang == 'spanish' else 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Preprocesamiento y tokenización
    def preprocess(examples):
        texts = [clean_text(t) for t in examples['message']]
        tok = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        # Mapear etiquetas mixtas
        tok['labels'] = [map_label(l) for l in examples['label']]
        return tok

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=['message', 'label'])
    ds_val   = ds_val.map(preprocess,   batched=True, remove_columns=['message', 'label'])

    # 5) Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
        save_total_limit=2
    )

    # 6) Data collator para padding dinámico
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7) Entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 8) Fine-tuning y evaluación
    trainer.train()
    metrics = trainer.evaluate()
    print("\nResultados validación:", metrics)

    # 9) Guardado
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()



