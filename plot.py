import matplotlib.pyplot as plt

# 数据准备：去除'valid'并整理成两列，一列是模型名称，一列是对应分数
data = [
    ("opt-350m", 19.48899482506002),
    ("checkpoint-10", 16.615849909147343),
    ("checkpoint-20", 16.109620501110886),
    ("checkpoint-30", 15.77740461697068),
    ("checkpoint-40", 15.847805161149028),
    ("checkpoint-50", 15.820256462067924),
    ("checkpoint-60", 15.738226713480639),
    ("checkpoint-70", 15.936774364917055),
    ("checkpoint-80", 15.833743006585788),
    ("checkpoint-90", 15.84002726871275),
    ("detected_model", 14.996239244658277),
]

# 分离模型名称和分数
model_names = [entry[0] for entry in data]
scores = [entry[1] for entry in data]

# 使用matplotlib绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(model_names, scores, color='skyblue')

# 添加标题和轴标签
plt.title('Model Perplexity in Clean Data')
plt.xlabel('Models')
plt.ylabel('Perplexity')

# 优化x轴的标签显示，避免重叠
plt.xticks(rotation=45, ha='right')
plt.ylim(13, 21)

# 显示图表
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()

plt.savefig('perplexity.png')
