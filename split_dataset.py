"""
split_dataset.py

Divide uno o varios CSVs de SMS en dos archivos: train_val y test,
manteniendo la proporción de clases.

Uso:
    python split_dataset.py \
        --data_paths data/esp_sms.csv data/eng_sms.csv \
        [--test_size 0.2] \
        [--random_state 42] \
        [--output_dir data]
"""
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def split_and_save(path, test_size, random_state, output_dir):
    # Carga del dataset completo
    df = pd.read_csv(path)
    # División manteniendo proporción de etiquetas
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    # Construcción de nombres de archivo
    base = os.path.splitext(os.path.basename(path))[0]
    train_val_path = os.path.join(output_dir, f"{base}_train.csv")
    test_path      = os.path.join(output_dir, f"{base}_test.csv")
    # Asegurar existencia del directorio
    os.makedirs(output_dir, exist_ok=True)
    # Guardado de los splits
    train_val.to_csv(train_val_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"✅ {base}:")
    print(f"    Train/val guardado en: {train_val_path}")
    print(f"    Test      guardado en: {test_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Split de uno o varios CSV de SMS en train_val y test sets."
    )
    parser.add_argument(
        '--data_paths', type=str, nargs='+', required=True,
        help='Rutas a los CSV originales (e.g. data/esp_sms.csv data/eng_sms.csv)'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help='Fracción del dataset para el test set (por defecto: 0.2)'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Semilla para reproducibilidad (por defecto: 42)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Directorio donde se guardarán los archivos resultantes (por defecto: data)'
    )
    args = parser.parse_args()

    for path in args.data_paths:
        split_and_save(
            path=path,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
