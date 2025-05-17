import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm


# 读取用户上传的文件
file_path = './all_names_combined.csv'

# 尝试读取文件内容，预览前几行
df = pd.read_csv(file_path)

# 显示列名和前几行，便于我们了解数据结构
print(df.head())





# 提取所有唯一的人名，并将它们转换为大写或小写（统一格式）
names = df['Name'].unique()
names = [name.strip().lower() for name in names if isinstance(name, str)]

# 构建字符集
all_text = ''.join(names)
chars = sorted(set(all_text))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

# 展示字符集及其大小
print(chars, len(chars))





import numpy as np

# 设置最大序列长度（例如最长前缀为10个字符）
max_seq_len = 10

# 构建训练数据集
sequences = []
next_chars = []

for name in names:
    for i in range(1, len(name)):
        seq = name[:i]  # 输入序列
        next_char = name[i]  # 目标字符
        sequences.append(seq)
        next_chars.append(next_char)
# 重新构造样本，保留长度在 [1, max_seq_len] 的序列
filtered_sequences = []
filtered_next_chars = []

for seq, next_char in zip(sequences, next_chars):
    if len(seq) <= max_seq_len:
        filtered_sequences.append(seq)
        filtered_next_chars.append(next_char)

# 重新构造输入矩阵 X 和目标向量 y
X = np.zeros((len(filtered_sequences), max_seq_len), dtype=np.int32)
y = np.zeros(len(filtered_sequences), dtype=np.int32)

for i, seq in enumerate(filtered_sequences):
    for t, char in enumerate(seq):
        X[i, max_seq_len - len(seq) + t] = char_to_idx[char]
    y[i] = char_to_idx[filtered_next_chars[i]]


print(X.shape, y.shape)




import torch
import torch.nn as nn

# 基本参数
vocab_size = len(char_to_idx)
embedding_dim = 32
hidden_dim = 128

# LSTM模型定义
class NamePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NamePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch_size, hidden_dim)
        output = self.fc(h_n.squeeze(0))  # (batch_size, vocab_size)
        return output

# 创建模型实例
model = NamePredictor(vocab_size, embedding_dim, hidden_dim)

# 查看模型结构
print(model)




# 设置训练设备（使用 GPU 如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# 构建数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


best_loss = float('inf')
save_path = 'best_name_model.pt'
epoch_losses = []




# 开始训练模型
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    total_samples = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        batch_size = batch_x.size(0)
        epoch_loss += loss.item() * batch_size  # 按样本数量加权
        total_samples += batch_size
        
        # 实时更新 tqdm 显示当前 batch loss
        pbar.set_postfix({"BatchLoss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / total_samples
    epoch_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved. New best loss: {best_loss:.6f}")

    print(f"\nEpoch {epoch}/{num_epochs}, Avg Loss per sample: {avg_loss:.6f}")

import matplotlib.pyplot as plt

plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Avg Loss per Sample')
plt.grid(True)
plt.show()



def predict_name(prefix, max_len=20):
    model.eval()
    prefix = prefix.lower()
    generated = prefix
    input_seq = [char_to_idx.get(ch, 0) for ch in prefix]  # 转换为索引序列

    for _ in range(max_len - len(prefix)):
        # 构造输入张量，右对齐补零
        x = torch.zeros((1, max_seq_len), dtype=torch.long).to(device)
        x[0, -len(input_seq):] = torch.tensor(input_seq[-max_seq_len:], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1).squeeze()  # 概率分布
            next_char_idx = torch.argmax(probs).item()      # greedy 选最大概率字符

        next_char = idx_to_char[next_char_idx]
        generated += next_char
        input_seq.append(next_char_idx)

        # 可选结束逻辑：常用结尾字母可终止
        if next_char in ['n', 'y', 'e'] and len(generated) > 4:
            break

    return generated

test_prefixes = ['j', 'a', 'mi', 'li', 'el', 'ch']
for prefix in test_prefixes:
    print(f"{prefix} ➜ {predict_name(prefix)}")
