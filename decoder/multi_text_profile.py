import csv
import pandas as pd
import openai
import time
import tiktoken
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 读取 CSV 文件
file_path = 'enhancement.csv'  # 替换为你的文件路径
output_file = 'result.csv'  # 替换为你的文件路径

client = OpenAI(
    api_key="",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# 计算 token 数量
def count_tokens(prompt, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

# 截断输入，确保不超过最大 token 限制
def truncate_input(prompt, msg, prompt2, max_input_tokens=64000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = count_tokens(prompt + prompt2, model)
    max_msg_tokens = max_input_tokens - prompt_tokens  # 计算最大可用 token
    msg_tokens = encoding.encode(msg)
    truncated_msg = encoding.decode(msg_tokens[:max_msg_tokens])  # 截断消息
    return prompt + truncated_msg + prompt2

# GPT 预测
def test(msg):
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

    adjusted_prompt = truncate_input(prompt, msg, "", max_input_tokens=64000)

    while True:
        try:
            completion = client.chat.completions.create(
                model="deepseek-v3-241226",
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
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(10)  # 等待一分钟后重试
        except openai.APITimeoutError:
            print("Request timed out. Retrying...")
            time.sleep(10)  # 等待10秒后重试
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)  # 等待10秒后重试

# 处理单个数据点
def process_row(row):
    summary = row['summary'] if pd.notna(row['summary']) else ''
    description = row['description'] if pd.notna(row['description']) else ''


    role_mapping = {0: "the creator is a Regular User",
                    1: "the creator is a Cross-application Developer",
                    2: "the creator is an Inner-application Developer"}

    msg = f"\ncreator id: {row['creator']}\nrole:{role_mapping.get(row['role'], 'Unknown')}\nfrequency: {row['creator_freq']}"

    msg += f"\nSummary: {summary}\nDescription: {description}\n"

    # 获取 GPT 预测
    predicted_resolution = test(msg)

    # 计算是否正确
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

# 读取 CSV 文件并筛选需要处理的数据
df = pd.read_csv(file_path)
df = df[df['fold'].astype(str) != '0']  # 跳过 fold 为 0 的数据
total_rows = len(df)

# 线程池并行处理
num_threads = 10  # 线程数
batch_size = 800  # 每批处理的数据量
processed_count = 0  # 计数器，记录已处理数据量

print(f"Total rows: {total_rows}")
# 按批次处理
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
                csv_writer.writeheader()  # 仅写一次表头

            for future in as_completed(future_to_row):
                result = future.result()
                if result:
                    csv_writer.writerow(result)
                    processed_count += 1  # 计数

                    print(f"Processed {processed_count}/{total_rows} records...")

    print(f"Finished batch {batch_start} to {batch_end}. Total processed: {processed_count}/{total_rows}")

print("All batches completed.")
