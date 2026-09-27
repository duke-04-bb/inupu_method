import sys
from pathlib import Path

# 把 src 目录加入搜索路径，这样 import config 能找到 src\config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from tqdm import tqdm
import config
from sklearn.model_selection import train_test_split
import jieba
##构建数据集函数
def builid_dataset(sentenses,word2idx):
    indexed_sentenses = [[word2idx.get(word, 0) for word in jieba.lcut(sentence)] for sentence in sentenses]
    dataset = []
    for sentence in tqdm(indexed_sentenses, desc='构建数据集'):
        for i in range(len(sentence)-config.context_size):
            input=sentence[i:i+config.context_size]
            target=sentence[i+config.context_size]
            dataset.append({"input": input, "target": target})
    return dataset
def process():
    print('开始处理数据')
    df = pd.read_json(config.raw_data_path/"synthesized_.jsonl", lines=True, orient='records').sample(frac=0.1)
    ##提取句子
    sentenses=[]
    for dialog in df['dialog']:
        for sentence in dialog:
            sentenses.append(sentence.split('：')[1])
    print(sentenses[0:10])
    ###划分训练集和测试集
    train, test = train_test_split(sentenses, test_size=0.2, random_state=42)
    ##切词
    vocab = set()
    for sentence in tqdm(train,desc='切词中'):
        vocab.update(jieba.lcut(sentence))
    ##构建词表
    vocablist = ['<unk>'] + list(vocab)
    print('词表大小:', len(vocablist))
    with open(config.models_data_path/'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocablist))
    word2idx = {word: idx for idx, word in enumerate(vocablist)}
    train_data = builid_dataset(train, word2idx)
    ##保存训练集
    pd.DataFrame(train_data).to_json(config.processed_data_path/'train.jsonl', orient='records', lines=True)
    ##保存测试集
    test_data = builid_dataset(test, word2idx)
    pd.DataFrame(test_data).to_json(config.processed_data_path/'test.jsonl', orient='records', lines=True)
    print('数据处理完成，训练集和测试集已保存')
if __name__=='__main__':
    process()
    