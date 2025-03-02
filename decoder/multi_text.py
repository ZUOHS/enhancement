import csv
import pandas as pd
import openai
import time
import tiktoken
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

file_path = '../data/enhancement.csv' 
output_file = 'result.csv'  

client = OpenAI(
    api_key="",
    base_url=""
)


def count_tokens(prompt, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def truncate_input(prompt, msg, prompt2, max_input_tokens=64000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = count_tokens(prompt + prompt2, model)
    max_msg_tokens = max_input_tokens - prompt_tokens 
    msg_tokens = encoding.encode(msg)
    truncated_msg = encoding.decode(msg_tokens[:max_msg_tokens])  
    return prompt + truncated_msg + prompt2


def test(msg):
    prompt = '''
        """Enhancement Report"""
        The Enhancement Report is a document used in the software development and maintenance process to describe and track functional improvements and enhancements to existing software systems. It details proposed improvements, expected effects, priorities, and related technical and business requirements.

        **Instructions:**
        Based on the provided summary and description, give a resolution of the enhancement. The resolution should be one of the following: 
        - INVALID 
        - DUPLICATE 
        - FIXED 
        - WONTFIX 
        - INCOMPLETE 
        - WORKSFORME 
        - EXPIRED 
        - MOVED 
        - INACTIVE

        **Conditions:**
        - Answer with one word.
        '''

    adjusted_prompt = truncate_input(prompt, msg, "", max_input_tokens=64000)

    while True:
        try:
            completion = client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system",
                     "content": "You are a poetic assistant, skilled in giving resolution of enhancement."},
                    {"role": "user",
                     "content": adjusted_prompt},
                ],
                temperature=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError:
            print(openai.RateLimitError)
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(10)  
        except openai.APITimeoutError:
            print("Request timed out. Retrying...")
            time.sleep(10)  



def process_row(row):
    summary = row['summary'] if pd.notna(row['summary']) else ''
    description = row['description'] if pd.notna(row['description']) else ''

    msg = f"\nSummary: {summary}\nDescription: {description}\n"


    predicted_resolution = test(msg)


    is_correct = (row['resolution'] == 'FIXED' and 'fixed' in predicted_resolution.lower()) or \
                 (row['resolution'] != 'FIXED' and 'fixed' not in predicted_resolution.lower())

    return {
        'id': row['id'],
        'summary': row['summary'],
        'description': row['description'],
        'resolution': row['resolution'],
        'predicted_resolution': predicted_resolution,
        'is_correct': is_correct
    }


df = pd.read_csv(file_path)
df = df[df['fold'].astype(str) == '1']  
total_rows = len(df)


num_threads = 10  
batch_size = 800  
processed_count = 0  

print(f"Total rows: {total_rows}")

for batch_start in range(0, total_rows, batch_size):
    batch_end = min(batch_start + batch_size, total_rows)
    batch_data = df.iloc[batch_start:batch_end]

    print(f"Processing batch {batch_start} to {batch_end}...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_row = {executor.submit(process_row, row): row for _, row in batch_data.iterrows()}

        with open(output_file, mode='a', encoding='utf-8', newline='') as result_file:
            fieldnames = ['id', 'summary', 'description', 'resolution', 'predicted_resolution', 'is_correct']
            csv_writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            if batch_start == 0:
                csv_writer.writeheader()

            for future in as_completed(future_to_row):
                result = future.result()
                if result:
                    csv_writer.writerow(result)
                    processed_count += 1  

                    print(f"Processed {processed_count}/{total_rows} records...")

    print(f"Finished batch {batch_start} to {batch_end}. Total processed: {processed_count}/{total_rows}")

print("All batches completed.")
