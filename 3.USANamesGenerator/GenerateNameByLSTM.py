import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ✅ 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ✅ 加载姓名数据
df = pd.read_csv("all_names_combined.csv")
names = df["Name"].dropna().unique().tolist()

# ✅ 构建字符集和映射
all_text = "\n".join(names)
chars = sorted(list(set(all_text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)

# ✅ 构建字符序列训练样本
sequences = []
for name in names:
    for i in range(1, len(name)):
        seq_in = name[:i]
        seq_out = name[i]
        sequences.append((
            [char2idx[c] for c in seq_in],
            char2idx[seq_out]
        ))

max_len = max(len(seq[0]) for seq in sequences)
X = np.zeros((len(sequences), max_len), dtype=np.int64)
y = np.zeros(len(sequences), dtype=np.int64)

for i, (seq_in, seq_out) in enumerate(sequences):
    X[i, -len(seq_in):] = seq_in
    y[i] = seq_out

# ✅ 数据拆分
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1)

# ✅ Dataset 封装
class NameDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(NameDataset(X_train, y_train), batch_size=256, shuffle=True)

# ✅ LSTM 模型定义
class NameLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)

# ✅ 初始化
model = NameLSTM(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

# ✅ 模型训练（50轮 + tqdm）
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=True,  # 保留进度条
                ncols=100,  # 设置进度条宽度
                bar_format='{l_bar}{bar:30}{r_bar}')  # 自定义进度条格式
    
    for batch_x, batch_y in loop:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(loop.n+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# ✅ Top-k 采样函数
def sample_next_char(output, top_k=5):
    probs = torch.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probs, k=top_k)
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()
    next_idx = np.random.choice(top_indices, p=top_probs / top_probs.sum())
    return next_idx

# ✅ 生成姓名函数（支持随机采样）
def generate_name(prefix='', max_len=10):
    model.eval()
    name = prefix
    input_seq = [char2idx.get(c, 0) for c in prefix]
    for _ in range(max_len):
        pad = [0] * (max_len - len(input_seq)) + input_seq
        x = torch.tensor([pad], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(x)
            next_idx = sample_next_char(output, top_k=5)
            next_char = idx2char[next_idx]
            name += next_char
            input_seq.append(next_idx)
    return name

# ✅ 示例生成
print("\n🌟 生成的新名字示例：")
for _ in range(10):
    print(generate_name(prefix='A'))
