# -*- coding:utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©2025丁师兄大模型
# 生成时间: 2025-05
# --------------------------------------------

"""
    usage: python run.py --model bert --data intent
"""

import time
import torch
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
from data_helper import build_dataset, build_iterator, get_time_dif
from utils import logger

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, bert_tiny')
parser.add_argument('--data', type=str, required=True, help='choose a dataset: reject, intent')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.data  # 数据集
    model_name = args.model  # 模型
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    # 设置随机数种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    logger.info("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logger.info(f"Time usage: {time_dif}")

    # train & eval
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
