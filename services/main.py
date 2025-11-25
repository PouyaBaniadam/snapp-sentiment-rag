import pandas as pd
import requests
import time
import os
import json
from tqdm import tqdm
import config



def get_last_processed_index(filename: str) -> int:
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        pd.DataFrame(columns=['original_index', 'predicted_label', 'model_used']).to_csv(filename, index=False)
        return -1
    try:
        df_results = pd.read_csv(filename)
        if df_results.empty:
            return -1
        return df_results['original_index'].max()
    except Exception as e:
        print(f"CRITICAL ERROR reading output file: {e}")
        print("Suggestion: Delete the output CSV file and restart.")
        exit()


def create_prompt(comments_batch: list) -> str:
    base_prompt = """
You are an expert sentiment analysis model for Persian comments.
Your task is to classify a list of comments about a food delivery service as either "HAPPY" or "SAD".
You MUST follow the specified input/output format precisely.

---
**EXAMPLE 1**
Input:
1. "غذا عالی بود، خیلی زود رسید"
2. "پیک خیلی تاخیر داشت."
Output:
["HAPPY", "SAD"]

---
**EXAMPLE 2**
Input:
1. "کیفیت غذا اصلا خوب نبود."
2. "مثل همیشه عالی."
Output:
["SAD", "HAPPY"]

---
Remember, your final list length must be exactly the same as input comments count.
---
Now, process the following comments and provide ONLY the JSON output.

**Input Comments:**
"""
    for i, comment in enumerate(comments_batch, 1):
        base_prompt += f'{i}. "{comment}"\n'
    return base_prompt


def create_batches(df: pd.DataFrame, char_limit: int):
    current_batch_comments = []
    current_batch_indices = []
    base_prompt_len = len(create_prompt([]))

    for _, row in df.iterrows():
        comment = row['comment']
        index = row['original_index']

        added_len = len(comment) + 10

        current_len = base_prompt_len + len("".join(current_batch_comments)) + (len(current_batch_comments) * 10)

        if (current_len + added_len) > char_limit and current_batch_comments:
            yield (current_batch_comments, current_batch_indices)
            current_batch_comments = []
            current_batch_indices = []

        current_batch_comments.append(comment)
        current_batch_indices.append(index)

    if current_batch_comments:
        yield (current_batch_comments, current_batch_indices)


def get_available_model():
    headers = {"Authorization": f"Bearer {config.JWT_TOKEN}"}
    try:
        response = requests.get(config.STATUS_CHECK_URL, headers=headers)
        response.raise_for_status()

        data = response.json()

        snapfood_data = None
        for result in data.get('results', []):
            if result.get('name') == 'Snapfood':
                snapfood_data = result
                break

        if not snapfood_data:
            print("ERROR: Project 'Snapfood' not found in API response.")
            return None, None

        models_data = snapfood_data.get('models_data', {})

        config_items = []
        for k, v in config.MODEL_CONFIG.items():
            config_items.append((k, v))

        config_items.sort(key=lambda x: x[1]['priority'])

        for model_name, model_info in config_items:
            if model_name in models_data:
                model_api_data = models_data[model_name]
                if 'plans' in model_api_data and len(model_api_data['plans']) > 0:
                    metrics = model_api_data['plans'][0]['metrics']

                    total_owned = metrics['total_requests']['owned']
                    total_called = metrics['total_requests']['called']
                    daily_owned = metrics['requests_per_day']['owned']
                    daily_called = metrics['requests_per_day']['called']

                    if total_called < total_owned and daily_called < daily_owned:
                        return model_name, model_info

        return None, None

    except requests.exceptions.RequestException as e:
        print(f"\nAPI CHECK ERROR: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"SERVER RESPONSE: {e.response.text}")
        return None, None
    except Exception as e:
        print(f"\nUNEXPECTED ERROR in get_available_model: {e}")
        return None, None


def call_llm_api(prompt: str, model: str):
    payload = {"model": model, "prompt": prompt}
    headers = {"Authorization": f"Bearer {config.API_KEY}"}

    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.post(config.GENERATE_ENDPOINT, data=payload, headers=headers)
            response.raise_for_status()

            initial_data = response.json()
            result_url = initial_data.get("result_url")

            if not result_url:
                print(f"ERROR: No 'result_url' returned. Response: {initial_data}")
                continue

            while True:
                time.sleep(config.BATCH_POLL_INTERVAL_SECONDS)
                res_resp = requests.get(result_url, headers=headers)
                res_resp.raise_for_status()

                res_data = res_resp.json()
                status = res_data.get("status")

                if status == 'Done':
                    text_answer = res_data.get("text_answer", "")
                    try:
                        start = text_answer.find('[')
                        end = text_answer.rfind(']') + 1
                        if start == -1 or end == 0:
                            raise ValueError("No JSON list found in text")

                        json_str = text_answer[start:end]
                        return json.loads(json_str)
                    except Exception as e:
                        print(f"\nJSON PARSING ERROR: {e}")
                        print(f"MODEL RAW OUTPUT: {text_answer}")
                        return None

                elif status == 'Error':
                    print(f"\nAPI PROCESSING ERROR (Attempt {attempt + 1}).")
                    break

        except requests.exceptions.HTTPError as http_err:
            print(f"\nHTTP ERROR (Attempt {attempt + 1}): {http_err}")
            print(f"SERVER RESPONSE BODY: {http_err.response.text}")
            time.sleep(5)

        except requests.exceptions.RequestException as e:
            print(f"\nNETWORK ERROR (Attempt {attempt + 1}): {e}")
            time.sleep(5)

    return None


if __name__ == "__main__":
    print("--- STARTING SENTIMENT ANALYSIS (Debug Mode) ---")

    if not os.path.exists(config.INPUT_FILE):
        print(f"ERROR: Input file not found at: {config.INPUT_FILE}")
        exit()

    df_main = pd.read_csv(config.INPUT_FILE)
    df_main['comment'] = df_main['comment'].astype(str)

    total_rows = len(df_main)
    progress_bar = tqdm(total=total_rows, desc="Total Progress")

    while True:
        last_idx = get_last_processed_index(config.OUTPUT_FILE)
        start_index = last_idx + 1

        progress_bar.n = start_index
        progress_bar.refresh()

        if start_index >= total_rows:
            print("\nAll comments processed successfully.")
            break

        model_name, model_info = get_available_model()

        if not model_name:
            print(f"\nNo available models found. Waiting {config.WAIT_SECONDS_IF_NO_MODEL}s...")
            time.sleep(config.WAIT_SECONDS_IF_NO_MODEL)
            continue

        df_subset = df_main.iloc[start_index:].copy()
        df_subset.reset_index(drop=True, inplace=True)
        df_subset['original_index'] = df_subset.index + start_index

        batch_gen = create_batches(df_subset, model_info['char_limit'])
        try:
            comments, indices = next(batch_gen)
        except StopIteration:
            print("\nNo data left to batch.")
            break

        progress_bar.set_description(f"Processing {indices[0]}-{indices[-1]} via {model_name}")

        prompt = create_prompt(comments)
        predicted_labels = call_llm_api(prompt, model_name)

        save_mode = 'a'

        print(predicted_labels)
        print(len(predicted_labels))
        print(len(indices))

        if predicted_labels and len(predicted_labels) == len(indices):
            status_labels = predicted_labels
        else:
            print(f"\nWARNING: Batch {indices[0]}-{indices[-1]} FAILED. Saving as error.")
            status_labels = ["PROCESSING_FAILED"] * len(indices)

        temp_df = pd.DataFrame({
            'original_index': indices,
            'predicted_label': status_labels,
            'model_used': model_name
        })

        temp_df.to_csv(config.OUTPUT_FILE, mode=save_mode, header=False, index=False)

        progress_bar.update(len(indices))

    progress_bar.close()
    print("\n--- SCRIPT FINISHED ---")