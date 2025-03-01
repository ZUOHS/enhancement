import csv
import pandas as pd
import openai
import time
import tiktoken
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

file_path = '../data/enhancement.csv' 
output_file = 'result.csv'  

client = OpenAI(
    api_key="",
    base_url=""
)

csv_lock = threading.Lock()


def count_tokens(prompt, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def truncate_input(example, prompt, msg, prompt2, max_input_tokens=64000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)

    prompt_tokens = count_tokens(prompt + prompt2, model)
    max_msg_tokens = max_input_tokens - prompt_tokens 

    msg_tokens = encoding.encode(msg)
    truncated_msg = encoding.decode(msg_tokens[:max_msg_tokens])  

    prompt_tokens2 = count_tokens(prompt + truncated_msg + prompt2, model)
    max_example_tokens = max_input_tokens - prompt_tokens2  
    example_tokens = encoding.encode(example)
    truncate_example = encoding.decode(example_tokens[:max_example_tokens])

    return truncate_example + prompt + truncated_msg + prompt2


def test(msg, example):
    prompt = '''
            """Enhancement Report"""
            The Enhancement Report is a document used in the software development and maintenance process to describe and track functional improvements and enhancements to existing software systems. It details proposed improvements, expected effects, priorities, and related technical and business requirements.

            **Instructions:**
            Based on the provided summary, description, and the profile of the creator, give a resolution of the enhancement. The resolution should be one of the following: 
            - INVALID 
            - DUPLICATE 
            - FIXED 
            - WONTFIX 
            - INCOMPLETE 
            - WORKSFORME 
            - EXPIRED 
            - MOVED 
            - INACTIVE

            **Role Descriptions:**
            - **Inner-application Developer:** Developer of this application and typically have deep insight into the codebase.
            - **Cross-application Developer:** Developer of another application.
            - **Regular User:** A non-developer who requests enhancements based on their experience using the software. 

            **Conditions:**
            - Answer with one word.
            '''

    adjusted_prompt = truncate_input(example, prompt, msg, "", max_input_tokens=64000)

    while True:
        try:
            completion = client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system",
                     "content": "You are a poetic assistant, skilled in giving resolution of enhancement."},
                    {"role": "user", "content": adjusted_prompt},
                ],
                temperature=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(10)
        except openai.APITimeoutError:
            print("Request timed out. Retrying...")
            time.sleep(10)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)


def process_row(row, df, roles):
    if row['fold'] == '0' or row['fold'] == 0:
        return None

    summary = row['summary'] if pd.notna(row['summary']) else ''
    description = row['description'] if pd.notna(row['description']) else ''
    role_name = roles[int(row['role'])]
    msg = f"\ncreator id: {row['creator']}{role_name}\nfrequency: {row['creator_freq']}\nSummary: {summary}\nDescription: {description}\n"

    fixed_examples = []
    non_fixed_examples = []
    total_rows = len(df)

    for i in range(1, 6):
        related_index = row.get(f'related_fixed_{i}')
        if pd.notna(related_index):
            try:
                idx = int(related_index)
            except Exception as e:
                continue
            if idx < 0 or idx >= total_rows:
                continue
            related_row = df.iloc[idx]
            fixed_examples.append(
                f"Example {i}:\ncreator id: {related_row['creator']}{roles[int(related_row['role'])]}\nfrequency: {related_row['creator_freq']}\nSummary: {related_row['summary']}\nResolution: {related_row['resolution']}\n"
            )

    for i in range(1, 6):
        related_index = row.get(f'related_non_fixed_{i}')
        if pd.notna(related_index):
            try:
                idx = int(related_index)
            except Exception as e:
                continue
            if idx < 0 or idx >= total_rows:
                continue
            related_row = df.iloc[idx]
            non_fixed_examples.append(
                f"Example {i}:\ncreator id: {related_row['creator']}{roles[int(related_row['role'])]}\nfrequency: {related_row['creator_freq']}\nSummary: {related_row['summary']}\nResolution: {related_row['resolution']}\n"
            )

    example_str = "\n".join(fixed_examples + non_fixed_examples)
    res = test(msg, example_str)
    if not res or res.strip() == "":
        return None

    is_correct = 'fixed' in res.lower() if row['resolution'] == 'FIXED' else 'fixed' not in res.lower()
    result = {
        'id': row['id'],
        'summary': row['summary'],
        'description': row['description'],
        'resolution': row['resolution'],
        'predicted_resolution': res,
        'is_correct': is_correct,
        'product': row['product'],
        'role': row['role']
    }
    with csv_lock:
        with open(output_file, mode='a', encoding='utf-8', newline='') as result_file:
            csv_writer = csv.DictWriter(result_file, fieldnames=['id', 'summary', 'description', 'resolution', 'predicted_resolution', 'is_correct', 'product', 'role'])
            csv_writer.writerow(result)
    return result



df = pd.read_csv(file_path)
df = df[df['fold'].astype(str) == '1']  
total_rows = len(df)
roles = ["\nrole:the creator is a Regular User", "\nrole:the creator is a Cross-application Developer", "\nrole:the creator is a Inner-application Developer"]
num_threads = 8
processed_count = 0  

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_row, df.iloc[i], df, roles) for i in range(len(df))]
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                processed_count += 1 

                print(f"Processed {processed_count}/{total_rows} records...")
                print(result['id'], result['resolution'], result['predicted_resolution'])
        except Exception as e:
            print(f"Error processing row: {e}")

print("Processing complete.")