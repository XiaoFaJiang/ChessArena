# 使用官方Python镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

COPY requirements.txt requirements.txt
# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件(忽略日志文件)
COPY . .

# 设置容器启动命令
CMD ["python", "elo_calculate.py"]