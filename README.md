
# 2025FudanBuqing · 循环神经网络项目实践

本项目为“2025复旦步青计划”中关于循环神经网络（RNN）应用的实训项目，分为三个子模块，分别展示了 **Simple RNN** 和 **LSTM** 在情感分析与字符生成任务中的不同实践。  

---

## 📁 文件夹结构概览

```
.
├── 1.simple-RNN-sentiment
│   └── 使用Simple RNN进行影评情感分析
├── 2.sentiment-analysis-IMDB-Review-using-LSTM
│   └── 使用LSTM进行IMDB影评情感分析
└── 3.name-generation-LSTM
    └── 使用LSTM进行美国人名生成
```

---

## 📦 模块一：Simple RNN 影评情感分析

### 🎯 任务描述  
使用 Simple RNN 模型对影评数据进行训练与预测，判断评论为“正面”或“负面”。

### 🛠️ 特点  
- 模型结构简单，训练速度快
- 能力有限，难以处理长依赖序列
- 适合作为序列建模入门练习

---

## 📦 模块二：LSTM 影评情感分析

### 🎯 任务描述  
使用 LSTM 模型对 [IMDB 影评数据集](https://ai.stanford.edu/~amaas/data/sentiment/) 进行情感分类。

### 📚 所需依赖  
位于 `requirements.txt` 中，包括：
```text
pandas
numpy
nltk
scikit-learn
tensorflow
```

### 🚀 实现步骤
1. **加载数据**：读取 IMDB 数据集（50K 条影评）。
2. **数据清洗**：去除 HTML、标点符号、停用词。
3. **标签编码**：将情感标记为 0（负面）或 1（正面）。
4. **数据划分**：拆分为训练集和测试集。
5. **分词与序列处理**：Tokenizer + pad_sequences。
6. **构建模型**：
    ```python
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=32, input_length=130))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```
7. **训练评估**：使用 `model.fit()` 训练并评估准确率。

### ✅ 效果说明  
相比 Simple RNN，LSTM 显著提高了情感分类的准确率，能够更有效捕捉句子中的长期依赖关系。

---

## 📦 模块三：LSTM 美国人名生成

### 🎯 任务描述  
构建一个基于字符级语言模型的 LSTM 模型，生成符合美式命名规律的英文名。

### 📋 训练流程
1. **数据加载**：
   - 从 CSV 文件中读取英文名字
   - 全部转换为小写，构建字符字典

2. **训练样本构建**：
   - 将每个名字拆分为字符序列
   - 构建输入张量 X（字符序列）与标签 y（下一个字符）

3. **模型训练**：
   - 使用 PyTorch 的 `nn.LSTM` 构建模型
   - 优化器：Adam（学习率=0.005）
   - 损失函数：CrossEntropyLoss
   - Batch size = 512，Epoch = 20
   - 每轮保存最优模型，并绘制 loss 曲线

### 📓 教学用 Jupyter Notebook  
该模块提供完整的 Notebook 文件，用于演示字符生成模型的训练过程，适合教学展示。

---

## 🔍 模型对比分析

| 模型        | 训练速度 | 精度表现 | 长期依赖处理能力 |
|-------------|-----------|-----------|-------------------|
| Simple RNN  | 较快      | 中等      | 弱                |
| LSTM        | 较慢      | 高        | 强                |

> LSTM 在实际表现上优于 Simple RNN，尤其是在需要理解句子上下文含义的自然语言任务中更为有效。

---

## 📌 项目说明

本项目为 2025 年复旦大学“步青计划”项目课程作品，旨在通过实践教学加深对 RNN 及 LSTM 网络的理解与应用，覆盖从文本分类到语言生成的典型任务场景。

欢迎同行使用、参考或提出建议！

---

如需 Jupyter Notebook 版本或模型权重文件，请查看对应文件夹内的 `.ipynb` 文件及 `checkpoints/` 文件夹。

---  
📧 **联系邮箱**：fudan-buqing-ai@edu.cn  
🏫 **项目组织方**：复旦大学 AI 教育组
