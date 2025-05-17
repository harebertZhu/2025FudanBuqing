import os
import pandas as pd

# 路径：根目录下的 names 文件夹
folder_path = './names'  # 或者写成绝对路径 '/mnt/data/names'
output_file = 'all_names_combined.csv'

# 创建一个空的 DataFrame 来保存所有数据
all_data = pd.DataFrame(columns=["Name", "Gender", "Count", "SourceFile"])

# 遍历文件夹中的每个 .txt 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, header=None, names=["Name", "Gender", "Count"])
            df["SourceFile"] = filename  # 记录来源文件名
            all_data = pd.concat([all_data, df], ignore_index=True)
        except Exception as e:
            print(f"无法读取文件 {filename}，错误：{e}")

# 保存为 CSV 文件
all_data.to_csv(output_file, index=False)
print(f"已将 {len(all_data)} 条姓名记录保存到 {output_file}")
