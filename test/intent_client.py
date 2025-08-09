# -*- coding:utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©2025丁师兄大模型
# 生成时间: 2025-05
# --------------------------------------------

import os
import requests
import uuid
import json
from tqdm import tqdm


URL = os.environ["INTENT_URL"]

def get_completion(query):
    headers = {'Content-Type': 'application/json'}
    data = {"query": query, "trace_id": str(uuid.uuid1())}
    response = requests.post(url=URL, headers=headers, data=json.dumps(data))
    return response.json()

if __name__ == '__main__':
    fd = open("../train/data/intent/test.txt")
    right = 0
    total = 0
    data = fd.readlines()
    for index in tqdm(range(len(data))):
        line = data[index]
        text, label = line.strip().split("\t")
        label = int(label)
        response = get_completion(text)
        # print(text, response)
        if int(response["data"].split(",")[0]) == label:
            right += 1
        total += 1
    print("test avg acc@1:", right/total)
