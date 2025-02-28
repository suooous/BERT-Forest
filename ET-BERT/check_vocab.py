# 读取词汇表
with open("models/encryptd_vocab.txt", 'r', encoding='utf-8') as f:
    vocab = f.readlines()
print("词汇表大小:", len(vocab))