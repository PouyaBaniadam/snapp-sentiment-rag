import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_PATH = os.path.join(BASE_DIR, 'data', 'train_results_combined.csv')


def evaluate_models():
    if not os.path.exists(COMBINED_PATH):
        print(f"Error: {COMBINED_PATH} not found.")
        print("Please run 'python services/combine.py' first.")
        return

    df = pd.read_csv(COMBINED_PATH)

    print("\n" + "=" * 40)
    print(f"{'MODEL PERFORMANCE REPORT':^40}")
    print("=" * 40)

    unique_models = df['model_used'].unique()

    for model in unique_models:
        subset = df[df['model_used'] == model]

        y_true = subset['label']
        y_pred = subset['predicted_label']

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nModel: {model}")
        print(f"-" * 20)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")

    print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    evaluate_models()