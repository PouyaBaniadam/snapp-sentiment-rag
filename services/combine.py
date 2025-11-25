import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
RESULTS_PATH = os.path.join(BASE_DIR, 'data', 'results.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'train_results_combined.csv')


def combine_data():
    print("--- 1. Loading Files ---")
    try:
        df_train = pd.read_csv(TRAIN_PATH)
        df_results = pd.read_csv(RESULTS_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("--- 2. Merging Data ---")
    merged_df = pd.merge(
        df_train,
        df_results,
        left_index=True,
        right_on='original_index',
        how='inner'
    )

    merged_df['is_correct'] = merged_df['label'] == merged_df['predicted_label']

    columns_to_keep = [
        'original_index',
        'comment',
        'label',
        'predicted_label',
        'is_correct',
        'model_used'
    ]

    final_cols = [c for c in columns_to_keep if c in merged_df.columns]
    final_df = merged_df[final_cols]

    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"--- Success! ---")
    print(f"Combined file saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    combine_data()