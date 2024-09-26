# 先构建前端
cd web & npm run build

# 后端依赖
pip install leptonai
#### 其他的看少什么就安装什么吧

# 运行后端
python fastapi_server.py
#### 尝试过使用 fastapi， 但涉及把类的实例方式加上 fastapi 的 app.get, aap.post 装饰器， 影响的代码太多， 放弃

# 使用
http://localhost:9876

# 配置
都在 fastapi_server.py 里前面几行

## 后端端口号
BACKEND_PORT = 9876
## 大语言模型服务地址
OPENAI_URL = "http://localhost:11434/v1"
## 大语言模型 api key
OPENAI_API_KEY = "123456"
## 模型名
OPENAI_MODEL = "kuqoi/qwen2-tools"
## 参考前多少条结果作为问答上下文
REFERENCE_COUNT = 8

# 说明
1. 前端输入问题
2. 调用 bing search api 查到若干结果， 取前 REFERENCE_COUNT 条结果的 snippet 字段作为和大模型问答的上下文
3. 拿到大模型的回答， 和引用的搜索结果返回给前端

# todo
相关问题 和 tool 应该是还没具体实现

# 可能的问题
如果自动跳转到 /ledoc 页面去了， 可能是前端 react 的路由有点问题， 在处理路由发生异常后的跳转， 但我对前端开发基本不懂， 目前看在 python 3.11 时会发生， python 3.12 正常， 但这似乎应该是前端， 不解。

---
# 构建镜像

cd web && npm run build
docker build -t local_lepton_search .

# run
docker run -p 9999:8080 -e BING_SEARCH_V7_SUBSCRIPTION_KEY=b618a618157049fd90b954b41f2581f0 local_lepton_search 

docker run -it --rm --entrypoint /bin/bash local_lepton_search