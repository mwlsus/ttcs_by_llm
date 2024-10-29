import argparse
import http.client
import json
import base64
import os
import pandas as pd
import ssl
import ast
import numpy as np
import pickle
import dashscope
from http import HTTPStatus

def parse_arguments():
    parser = argparse.ArgumentParser(description="大语言模型乒乓教练")

    parser.add_argument('--api_key', type=str, required=True, help='API 密钥')
    parser.add_argument('--base_url', type=str, required=True, help='API 基础 URL')
    parser.add_argument('--model', type=str, required=True, help='使用的模型名称')
    parser.add_argument('--fpath', type=str, required=True, help='工作文件夹路径')
    parser.add_argument('--type', type=int, choices=range(1, 9), required=True, help='类型，1-8的整数')
    parser.add_argument('--kb_file', type=str, required=True, help='知识库文件的路径')
    parser.add_argument('--pdict', type=str, required=True, help='prompt字典')
    parser.add_argument('--ext', type=str, default='', help='附加内容')
    
    return parser.parse_args()

args = parse_arguments()
api_key = args.api_key
BASE = args.base_url

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_r(messages,api_key,model='gpt-4-turbo',temperature=0.5):

    context = ssl._create_unverified_context()
    conn = http.client.HTTPSConnection(BASE,context=context)
    payload = json.dumps({
    "model": model,
    "temperature": temperature,
    "messages": messages,
    "max_tokens":4096,
    })
    headers = {
    'Authorization': 'Bearer {}'.format(api_key),
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    #print(json.loads(res.read().decode('utf-8')))
    r = json.loads(res.read().decode('utf-8'))
    print(r)
    resText =  r['choices'][0]['message']['content']
    print(resText)
    return resText
    

def get_embedding(input,api_key):
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


def get_kb(fpath,prompt_kb):
    
    image1 = encode_image(r'{}\people.jpg'.format(fpath))
    image2 = encode_image(r'{}\paddle.jpg'.format(fpath))
    image3 = encode_image(r'{}\pptrace.jpg'.format(fpath))

        
    ml =  [{"role":'user',
    "content":[
    {
    "type": "text",
    "text": prompt_kb,
    },{"type": "image_url",
    "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
    },{"type": "image_url",
    "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
    },{"type": "image_url",
    "image_url":  {'url' : f"data:image/jpeg;base64,{image3}"}
    }

    ]}]
                    
    r = get_r( ml,api_key,'gpt-4-turbo')
    print('success_kb')
    remb =get_embedding(r,api_key)
    remb = np.array(remb)
    # 计算每个向量和 remb 的欧氏距离
    distances = np.linalg.norm(embs - remb, axis=1)
    
    # 找到距离最近的3个向量的索引
    nearest_indices = np.argsort(distances)[:3] + 1 
    kbdf = df[df['index'].isin(nearest_indices.tolist())]
    kbdfq = kbdf['question'].to_list()
    kbdfa = kbdf['answer'].to_list()
    kb = ''
    for i in range(len(kbdfq)):
        kb =kb +  f'Q{i}:{kbdfq[i]} \nA{i}:{kbdfa[i]} \n'
    return r


def get_error_1(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\arm_vector.jpg'.format(fpath))
        image2 = encode_image(r'{}\paddle.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text":prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
        ]}]             
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_1(fpath,model,prompt,ext,kb)

def get_error_2(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\arm_vector.jpg'.format(fpath))
        image2 = encode_image(r'{}\arm_angle.jpg'.format(fpath))
        image3 = encode_image(r'{}\people.jpg'.format(fpath))

            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image3}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_2(fpath,model,prompt,ext,kb)

def get_error_3(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\pptrace.jpg'.format(fpath))
        image2 = encode_image(r'{}\paddle.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_3(fpath,model,prompt,ext,kb)

def get_error_4(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\pptrace.jpg'.format(fpath))
        image2 = encode_image(r'{}\paddle.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_4(fpath,model,prompt,ext,kb)
def get_error_5(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\pptrace.jpg'.format(fpath))
        image2 = encode_image(r'{}\paddle.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_5(fpath,model,prompt,ext,kb)
        
def get_error_6(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\bcx.jpg'.format(fpath))

            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_6(fpath,model,prompt,ext,kb)

def get_error_7(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\bcx.jpg'.format(fpath))
        image2 = encode_image(r'{}\shoulder_shake.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_7(fpath,model,prompt,ext,kb)

def get_error_8(fpath,model,prompt,ext,kb):
    try:
        image1 = encode_image(r'{}\bcx.jpg'.format(fpath))
        image2 = encode_image(r'{}\bcy.jpg'.format(fpath))
            
        ml =  [{"role":'user',
        "content":[
        {
        "type": "text",
        "text": prompt.replace('$ext',ext).replace('$kb',kb),
        },{"type": "image_url",
        "image_url": {'url' : f"data:image/jpeg;base64,{image1}"}
        },{"type": "image_url",
        "image_url":  {'url' : f"data:image/jpeg;base64,{image2}"}
        }
    
        ]}]
                        
        r = get_r( ml,api_key,model)
        print('success',fpath)
        return r
    except:
        print('retry',fpath)
        return get_error_8(fpath,model,prompt,ext,kb)

    
if __name__ == "__main__":
    args = parse_arguments()
    print(f"API Key: {args.api_key}")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"工作文件夹路径: {args.fpath}")
    print(f"类型: {args.type}")
    print(f"知识库文件路径: {args.kb_file}")
    print(f"附加内容: {args.ext}")
    print(f"prompt字典：{args.pdict}")
    
    with open(args.pdict, 'rb') as file:
        # 使用 pickle.load() 函数从文件中反序列化对象
        pdata = pickle.load(file)
    if args.kb_file!='0':
        df = pd.read_csv(args.kb_file)
        embs = []
        for i in range(df.shape[0]):
           embs.append(ast.literal_eval(df[df['index']==i+1]['embedding'].to_list()[0]))
        embs = np.array(embs)
        kb = get_kb(args.fpath,pdata['kb'])
    else:
        kb = ''

    prompt = pdata[args.type]
    BASE=args.base_url
    api_key = args.api_key
    f = [print,get_error_1,get_error_2,get_error_3,get_error_4,get_error_5,get_error_6,get_error_7,get_error_8]
    r = f[args.type](args.fpath,args.model,prompt,args.ext,kb)

    print(r)
  