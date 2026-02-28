# 构建前请先执行: bash prepare_models.sh （生成 model_repository）
# 模型与客户端代码打入镜像，运行时无需挂载
ARG TRITON_BASE=registry.cn-heyuan.aliyuncs.com/leo03w/tritonserver:25.10-py3
FROM ${TRITON_BASE}

WORKDIR /app

# 模型仓库 → 运行时 /models
COPY model_repository /models

# 客户端/相关代码 → 便于容器内调试或二次调用
COPY client /app/client
COPY requirements-client.txt /app/

# 可选：容器内安装客户端依赖（按需取消注释）
# RUN pip install --no-cache-dir -r /app/requirements-client.txt

ENV MODEL_REPOSITORY=/models
EXPOSE 8000 8001 8002

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=false", \
     "--log-verbose=1"]
