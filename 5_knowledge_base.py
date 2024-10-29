import argparse
import http.client
import json
import base64
import os
import pandas as pd
import ssl

def get_embedding(input,api_key,BASE):
    context = ssl._create_unverified_context()
    conn = http.client.HTTPSConnection(BASE,context=context)
    payload = json.dumps(
    {
      "input": input,
      "model": "text-embedding-3-small"
    }
    )
    headers = {
    'Authorization': 'Bearer {}'.format(api_key),
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/embeddings", payload, headers)
    res = conn.getresponse()
    r = json.loads(res.read().decode('utf-8'))
    resText =  r['data'][0]['embedding']
    return resText
    
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='知识库生成程序，包含api_key, base_url和path_raw_kb参数')

# 添加参数
parser.add_argument('--api_key', type=str, required=True, help='API Key 用于身份验证')
parser.add_argument('--base_url', type=str, required=True, help='API 请求的基础 URL')
parser.add_argument('--path_raw_kb', type=str, required=True, help='原始知识库文件的路径')
parser.add_argument('--path_save_kb', type=str, required=True, help='保存知识库文件的路径')


# 解析参数
args = parser.parse_args()

# 打印参数值（用于调试）
print("API Key:", args.api_key)
print("Base URL:", args.base_url)
print("Path to Raw KB:", args.path_raw_kb)
print("Path to Save KB:", args.path_save_kb)


if __name__ == '__main__':
    api_key = args.api_key
    BASE = args.base_url
    # 打开并读取文件内容
    with open(args.path_raw_kb, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # 将数据分割为单独的问题和答案
    entries = data.strip().split('\n')
    kbase = {
    }
    
    # 分析每个问题和答案，创建字典
    for entry in entries:
        if entry.startswith('Q'):
            q_index = entry.split(': ')[0]
            question = entry.split(': ')[1]
            a_index = 'A' + q_index[1:]
            answer = next((e.split(': ')[1] for e in entries if e.startswith(a_index)), None)
            kbase[q_index] = {'index': q_index, 'question': question, 'answer': answer}
    
    new_kb = {
    'index':[],
    'question':[],
    'answer':  [],
    'embedding' : [],
    }
    for i in kbase:
        print(i)
        idx = int(kbase[i]['index'][1:])
        new_kb['index'].append(idx)
        new_kb['question'].append(kbase[i]['question'])
        new_kb['answer'].append(kbase[i]['answer'])
        new_kb['embedding'].append(get_embedding(kbase[i]['question'],api_key,BASE))
    
    
    df_n = pd.DataFrame(new_kb)
    df_n.to_csv(args.path_save_kb, index=False)