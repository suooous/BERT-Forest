import os
import shutil

# 创建目录
os.makedirs('datasets/test_data/packet', exist_ok=True)

# 测试数据内容 - 使用词汇表中的简单token
content = '''text_a	label
16 03 03 00 01	0
14 03 03 00 01	1
16 03 03 00 01	0
14 03 03 00 01	1
16 03 03 00 01	0'''.strip()

# 写入训练集
with open('datasets/test_data/packet/train_dataset.tsv', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
    if not content.endswith('\n'):
        f.write('\n')

# 复制为验证集和测试集
shutil.copy('datasets/test_data/packet/train_dataset.tsv', 'datasets/test_data/packet/valid_dataset.tsv')
shutil.copy('datasets/test_data/packet/train_dataset.tsv', 'datasets/test_data/packet/test_dataset.tsv')

print("Test data files created successfully!")

# 验证文件内容
for file in ['train_dataset.tsv', 'valid_dataset.tsv', 'test_dataset.tsv']:
    print(f"\nContents of {file}:")
    with open(f'datasets/test_data/packet/{file}', 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
        lines = content.strip().split('\n')
        print(f"Number of lines: {len(lines)}") 