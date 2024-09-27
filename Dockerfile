# 第一阶段， 使用官方的 Node.js 镜像作为基础镜像来构建前端
FROM node:22.0.0-slim AS build-stage

# 设置工作目录
WORKDIR /app

# 复制前端项目的
COPY ./web/public ./web/public
COPY ./web/src ./web/src
COPY ./web/*.js ./web/
COPY ./web/*.json ./web/
COPY ./web/*.ts ./web/
COPY ./web/*.mjs ./web/

# 构建前端项目
# RUN npm config set registry https://registry.npmmirror.com
# RUN npm install -g cnpm --registry=https://registry.npm.taobao.org
# RUN cd web && npm install && npm run build

WORKDIR /app
COPY ./ui ./ui

# 第二阶段， 使用官方的 Python 镜像作为基础镜像来运行后端
FROM python:3.12.5-slim

# 设置工作目录
WORKDIR /app

# 把第一阶段构建的结果复制过来
COPY --from=build-stage /app/ui ./ui

# 复制后端项目的所有 Python 文件
COPY ./*.py ./
COPY ./requirements.txt ./

# 安装后端项目的依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 暴露后端服务的端口
EXPOSE 8080

# 设置容器启动时执行的命令
CMD ["python", "fastapi_server.py"]