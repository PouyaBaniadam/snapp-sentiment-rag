import pandas as pd
import time
import os
from tqdm import tqdm

import main
import config


def retry_failed_batches():
    print("--- STARTING ERROR RETRY PROCESS ---")

    if not os.path.exists(config.OUTPUT_FILE):
        print(f"ERROR: {config.OUTPUT_FILE} not found. Run main.py first.")
        return

    df_results = pd.read_csv(config.OUTPUT_FILE)

    failed_mask = (df_results['predicted_label'] == 'PROCESSING_FAILED') | (df_results['predicted_label'].isna())
    failed_indices = df_results[failed_mask]['original_index'].values

    if len(failed_indices) == 0:
        print("No 'PROCESSING_FAILED' entries found. Results file looks clean!")
        return

    print(f"Found {len(failed_indices)} failed entries. preparing to process...")

    if not os.path.exists(config.INPUT_FILE):
        print(f"ERROR: Input file not found at: {config.INPUT_FILE}")
        exit()

    df_train = pd.read_csv(config.INPUT_FILE)
    df_train['comment'] = df_train['comment'].astype(str)

    try:
        df_failed = df_train.iloc[failed_indices].copy()
    except IndexError:
        print("ERROR: Index mismatch. The indices in results.csv exceed rows in train.csv.")
        return

    df_failed['original_index'] = df_failed.index
    df_processing_queue = df_failed.reset_index(drop=True)

    total_failed = len(df_processing_queue)
    pbar = tqdm(total=total_failed, desc="Retrying Errors")

    current_idx = 0

    while current_idx < total_failed:

        model_name, model_info = main.get_available_model()

        if not model_name:
            print(f"\nNo available models found. Waiting {config.WAIT_SECONDS_IF_NO_MODEL}s...")
            time.sleep(config.WAIT_SECONDS_IF_NO_MODEL)
            continue

        df_remaining = df_processing_queue.iloc[current_idx:].copy()

        batch_gen = main.create_batches(df_remaining, model_info['char_limit'])

        try:
            comments, indices = next(batch_gen)
        except StopIteration:
            break

        pbar.set_description(f"Retrying {indices[0]}-{indices[-1]} via {model_name}")

        prompt = main.create_prompt(comments)
        predicted_labels = main.call_llm_api(prompt, model_name)

        if predicted_labels and len(predicted_labels) == len(indices):
            for original_idx, label in zip(indices, predicted_labels):
                df_results.loc[df_results['original_index'] == original_idx, 'predicted_label'] = label
                df_results.loc[df_results['original_index'] == original_idx, 'model_used'] = model_name

            print(f" -> Batch {indices[0]}-{indices[-1]} recovered successfully.")
        else:
            print(f" -> Batch {indices[0]}-{indices[-1]} FAILED AGAIN.")

        df_results.to_csv(config.OUTPUT_FILE, index=False)

        processed_count = len(indices)
        current_idx += processed_count
        pbar.update(processed_count)

    pbar.close()
    print("\n--- RETRY PROCESS FINISHED ---")

    remaining_errors = len(df_results[df_results['predicted_label'] == 'PROCESSING_FAILED'])
    if remaining_errors == 0:
        print("SUCCESS: All errors have been fixed!")
    else:
        print(f"WARNING: {remaining_errors} entries still failed. You may run this script again.")


if __name__ == "__main__":
    retry_failed_batches()