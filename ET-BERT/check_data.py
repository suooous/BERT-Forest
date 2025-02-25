import pandas as pd

# 读取训练数据
df = pd.read_csv("datasets/CSTNET-TLS 1.3/packet/train_dataset.tsv", sep='\t')

# 检查每行的token数量
def count_tokens(text):
    return len(str(text).split())

# 计算统计信息
token_counts = df['text_a'].apply(count_tokens)
print("最大token数:", token_counts.max())
print("平均token数:", token_counts.mean())
print("样本数量:", len(df))
print("标签数量:", df['label'].nunique())   