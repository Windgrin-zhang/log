# 🚀 一周速成学习指南

## 学习原则
- **实践为主**：每学一个概念立即动手练习
- **项目驱动**：通过小项目巩固知识
- **重点突破**：掌握核心概念，细节后续补充

---

## 📅 第1天：Git + GitHub + SSH

### 上午：Git基础
**学习目标：**
- Git核心概念（工作区、暂存区、仓库）
- 基本命令：init, add, commit, log, status
- 分支管理：branch, checkout, merge

**实践项目：**
```bash
# 创建个人项目仓库
mkdir my-learning-project
cd my-learning-project
git init
echo "# 我的学习笔记" > README.md
git add README.md
git commit -m "初始提交"
```

**学习资源：**
- [Git官方教程](https://git-scm.com/book/zh/v2)
- [GitHub Guides](https://guides.github.com/)

### 下午：GitHub + SSH
**学习目标：**
- GitHub账号设置
- SSH密钥生成和配置
- 远程仓库操作：clone, push, pull

**实践项目：**
```bash
# 生成SSH密钥
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 添加远程仓库
git remote add origin git@github.com:username/repo.git
git push -u origin main
```

---

## 📅 第2天：Ubuntu Shell + 环境搭建

### 上午：Linux基础命令
**学习目标：**
- 文件操作：ls, cd, pwd, mkdir, rm, cp, mv
- 文本处理：cat, grep, sed, awk
- 权限管理：chmod, chown

**实践项目：**
```bash
# 创建学习目录结构
mkdir -p ~/learning/{git,ml,nlp,docker}
cd ~/learning
ls -la
```

### 下午：Jupyter Notebook
**学习目标：**
- Jupyter安装和启动
- Notebook基本操作
- Markdown和代码单元格

**实践项目：**
```bash
# 安装Jupyter
pip install jupyter notebook

# 启动Jupyter
jupyter notebook
```

**创建第一个Notebook：**
```python
# 测试环境
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("环境配置成功！")
```

---

## 📅 第3天：机器学习基础

### 上午：scikit-learn
**学习目标：**
- 机器学习基本概念
- 数据预处理
- 简单模型训练

**实践项目：**
```python
# 鸢尾花分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, predictions):.2f}")
```

### 下午：Keras深度学习
**学习目标：**
- 神经网络基础概念
- Keras API使用
- 简单神经网络构建

**实践项目：**
```python
# MNIST手写数字识别
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 📅 第4天：数据集平台 + Docker基础

### 上午：Kaggle + Hugging Face
**学习目标：**
- Kaggle账号注册和数据集下载
- Hugging Face模型和数据集使用
- 数据预处理技巧

**实践项目：**
```python
# 使用Hugging Face数据集
from datasets import load_dataset

# 加载情感分析数据集
dataset = load_dataset("sst2")
print(dataset['train'][0])

# 使用预训练模型
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
```

### 下午：Docker基础
**学习目标：**
- Docker概念和优势
- 基本命令：run, build, images, containers
- 简单Dockerfile编写

**实践项目：**
```dockerfile
# 创建简单的Python环境Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
# Docker基本操作
docker build -t my-ml-app .
docker run -p 8000:8000 my-ml-app
```

---

## 📅 第5天：NLP + 网络基础

### 上午：自然语言处理
**学习目标：**
- NLP基本概念
- 中英文处理差异
- HanLP工具使用

**实践项目：**
```python
# 使用HanLP进行中文处理
import hanlp

# 分词
tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
text = "我爱北京天安门"
tokens = tokenizer(text)
print(tokens)

# 命名实体识别
ner = hanlp.load('MSRA_NER_ELECTRA_BASE_ZH')
entities = ner(text)
print(entities)
```

### 下午：网络技术基础
**学习目标：**
- 网络基本概念
- QoS（服务质量）
- SMB/NAS文件共享

**实践项目：**
```bash
# 网络诊断
ping google.com
traceroute google.com
netstat -tuln

# SMB连接（Windows）
net use Z: \\server\share
```

---

## 📅 第6天：计算机视觉 + 流媒体

### 上午：计算机视觉基础
**学习目标：**
- 图像处理基本概念
- OpenCV基础操作
- 简单图像处理

**实践项目：**
```python
import cv2
import numpy as np

# 读取和显示图像
img = cv2.imread('image.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)

# 图像预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
```

### 下午：流媒体技术
**学习目标：**
- 流媒体基本概念
- RTSP协议
- 视频流处理

**实践项目：**
```python
# 简单的视频流处理
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📅 第7天：综合项目 + 复习

### 上午：综合项目
**项目目标：** 创建一个完整的机器学习项目

**项目内容：**
1. 使用Git管理代码
2. 在Jupyter中开发
3. 使用scikit-learn训练模型
4. 用Docker部署
5. 处理中文文本数据

### 下午：知识整理
- 复习一周学习内容
- 整理学习笔记
- 制定后续学习计划

---

## 🎯 每日检查清单

### 第1天
- [ ] Git基本命令熟练使用
- [ ] GitHub仓库创建和推送
- [ ] SSH密钥配置成功

### 第2天
- [ ] Linux常用命令掌握
- [ ] Jupyter环境搭建完成
- [ ] 第一个Notebook创建

### 第3天
- [ ] scikit-learn简单模型训练
- [ ] Keras神经网络构建
- [ ] 模型评估和优化

### 第4天
- [ ] Kaggle数据集下载使用
- [ ] Hugging Face模型调用
- [ ] Docker容器运行

### 第5天
- [ ] HanLP中文处理
- [ ] 网络基础概念理解
- [ ] SMB文件共享配置

### 第6天
- [ ] OpenCV图像处理
- [ ] 视频流捕获和显示
- [ ] 流媒体概念理解

### 第7天
- [ ] 综合项目完成
- [ ] 学习笔记整理
- [ ] 后续计划制定

---

## 💡 学习技巧

1. **番茄工作法**：25分钟专注学习 + 5分钟休息
2. **费曼学习法**：学完立即向他人解释
3. **实践优先**：理论结合实践，动手操作
4. **记录笔记**：及时记录学习心得和问题
5. **寻求帮助**：遇到问题及时搜索或请教

---

## 📚 推荐资源

### 在线教程
- [菜鸟教程](https://www.runoob.com/)
- [W3Schools](https://www.w3schools.com/)
- [Real Python](https://realpython.com/)

### 视频教程
- B站：搜索相关技术关键词
- YouTube：英文教程资源丰富

### 实践平台
- [Kaggle](https://www.kaggle.com/)
- [Hugging Face](https://huggingface.co/)
- [GitHub](https://github.com/)

---

**记住：** 一周学会这么多内容确实有挑战，但通过合理规划和高效学习，你一定能够掌握这些核心技能！加油！🚀 