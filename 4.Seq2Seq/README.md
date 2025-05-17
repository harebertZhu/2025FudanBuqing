# Seq2Seq 英德翻译项目

这是一个基于Seq2Seq模型的英德翻译项目，使用深度学习框架实现字符级别的机器翻译。

## 项目结构
Seq2Seq/
├── deu-eng/                # 英德平行语料库
│   ├── 新建 文本文档.txt   # 包含英德句子对
├── kor-eng/                # 韩英平行语料库
│   ├── _about.txt          # 数据来源说明
├── Seq2Seq.ipynb           # 基础版Seq2Seq实现(CPU)
├── Seq2Seq-GPU.ipynb       # GPU加速版Seq2Seq实现
├── char_level_seq2seq_translation.ipynb  # 字符级Seq2Seq实现
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明


## 数据集

- 数据来源：Tatoeba项目 (CC-BY 2.0许可)
- 包含英德平行句子对，格式为：`英文句子\t德文句子\t来源信息`

## 环境要求

- Python 3.x
- 依赖库：
  - numpy
  - torch
  - tensorflow
  - tqdm

安装依赖：
```bash
pip install -r requirements.txt


## 使用方法
1. 基础版(CPU): 运行 Seq2Seq.ipynb
2. GPU加速版: 运行 Seq2Seq-GPU.ipynb
3. 字符级实现: 运行 char_level_seq2seq_translation.ipynb
## 模型特点
- 基于LSTM的编码器-解码器架构
- 支持字符级别和单词级别的翻译
- 包含训练、验证和预测功能
- 模型保存与加载功能
## 许可
本项目数据遵循CC-BY 2.0许可，代码部分可根据需要自行选择许可方式。

## 注意事项
- 训练可能需要较长时间，建议使用GPU加速
- 模型性能取决于训练数据量和超参数设置
- 翻译质量可能受限于简单的Seq2Seq架构