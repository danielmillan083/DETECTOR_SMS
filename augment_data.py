"""
augment_data.py

Aumenta tu dataset de SMS en español mediante:
  1) Back‐translation: Español → Inglés → Español
  2) Traducción simple: Inglés → Español (sin duplicados)

Uso:
    python augment_data.py \
        --esp_train data/esp_sms_train_val.csv \
        --eng_train data/eng_sms_train_val.csv \
        --output_path data/esp_sms_augmented.csv \
        [--num_bt 1] \
        [--num_en2es 1] \
        [--verbose]

Dependencias:
    pip install pandas deep-translator tqdm
"""
import pandas as pd
import argparse
import os
from deep_translator import GoogleTranslator
from tqdm import tqdm

def augment_spanish(df, num_bt, verbose=False):
    """Genera variantes back-translation para dataframe en español."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Back-translation", disable=not verbose):
        message, label = row['message'], row['label']
        for _ in range(num_bt):
            try:
                en = GoogleTranslator(source='es', target='en').translate(message)
                es_bt = GoogleTranslator(source='en', target='es').translate(en)
                rows.append({'message': es_bt, 'label': label})
            except Exception as e:
                print(f"[Error BT] \"{message}\": {e}")
    return pd.DataFrame(rows)

def augment_english(df, num_en2es, verbose=False):
    """Genera variantes traduciendo del inglés al español, eliminando duplicados."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eng→Es", disable=not verbose):
        message, label = row['message'], row['label']
        translations = set()
        # Generar traducciones y añadir solo únicas
        for _ in range(num_en2es):
            try:
                es = GoogleTranslator(source='en', target='es').translate(message)
                translations.add(es)
            except Exception as e:
                print(f"[Error E→S] \"{message}\": {e}")
        # Añadir cada variante única
        for es_text in translations:
            rows.append({'message': es_text, 'label': label})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Data augmentation para SMS")
    parser.add_argument('--esp_train',    type=str, required=True,
                        help='CSV train_val en español')
    parser.add_argument('--eng_train',    type=str, required=True,
                        help='CSV train_val en inglés')
    parser.add_argument('--output_path',  type=str, required=True,
                        help='Ruta de salida para el CSV aumentado')
    parser.add_argument('--num_bt',       type=int, default=1,
                        help='Número de back‐translations por mensaje (esp)')
    parser.add_argument('--num_en2es',    type=int, default=1,
                        help='Número de traducciones en→es únicas por mensaje (eng)')
    parser.add_argument('--verbose',      action='store_true',
                        help='Mostrar barra de progreso')
    args = parser.parse_args()

    # Leer datos originales
    df_esp = pd.read_csv(args.esp_train)
    df_eng = pd.read_csv(args.eng_train)

    # Renombrar columna 'text' a 'message' si existe
    if 'text' in df_esp.columns:
        df_esp = df_esp.rename(columns={'text': 'message'})
    if 'text' in df_eng.columns:
        df_eng = df_eng.rename(columns={'text': 'message'})

    # Marcar ejemplos originales
    df_esp['aug_type'] = 'original_es'
    df_eng['aug_type'] = 'original_en'

    # Generar aumentos
    bt_df    = augment_spanish(df_esp, args.num_bt, args.verbose)
    en2es_df = augment_english(df_eng, args.num_en2es, args.verbose)

    # Concatenar todo
    df_aug = pd.concat([
        df_esp[['message','label','aug_type']],
        bt_df,
        en2es_df
    ], ignore_index=True)

    # Fusionar posible columna 'text' y limpiar
    if 'text' in df_aug.columns:
        df_aug['message'] = df_aug['message'].fillna(df_aug['text'])
        df_aug = df_aug[['message','label','aug_type']]

    # Guardar CSV aumentado
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_aug.to_csv(args.output_path, index=False)
    print(f"📈 Dataset aumentado guardado en: {args.output_path}")
    print(f"   Originales ES: {len(df_esp)}, BT: {len(bt_df)}, Eng→Es únicas: {len(en2es_df)}")

if __name__ == '__main__':
    main()
