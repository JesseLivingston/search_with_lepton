from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

import concurrent.futures
import glob
import json
import os
import re
import threading
import requests
import traceback
from typing import Annotated, List, Generator, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
import httpx
from loguru import logger

from leptonai.photon import Photon, StaticFiles
from leptonai.photon.types import to_bool
from leptonai.util import tool

# 
BACKEND_PORT = 8080

OPENAI_URL = "http://localhost:11434/v1"
# OPENAI_URL = "http://host.docker.internal:11434/v1"
OPENAI_API_KEY = "123456"
OPENAI_MODEL = "kuqoi/qwen2-tools" # "llama3:latest"

# api key
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_SEARCH_V7_SUBSCRIPTION_KEY = ""

BING_MKT = "zh-CN"
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5

# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5


# If the user did not provide a query, we will use this default query.
_default_query_en = "Who said 'live long and prosper'?"
_default_query = "问苍茫大地， 谁主沉浮？"

# This is really the most important part of the rag model. It gives instructions
# to the model on how to generate the answer. Of course, different models may
# behave differently, and we haven't tuned the prompt to make it optimal - this
# is left to you, application creators, as an open problem.
_rag_query_text_en = """
You are a large language AI assistant built by Lepton AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""
_rag_query_text = """
你是由 CCDC AI 构建的大型语言 AI 助手。你将收到一个用户问题，请写出简洁、准确且清晰的答案。你会得到一组与问题相关的上下文，每个上下文都以引用编号开头，例如
[[citation:x]]，其中x是一个数字。如果适用，请在每句话的末尾引用上下文。

你的回答必须是正确、准确且由专家使用公正和专业的语气撰写的。请限制在1024个token以内。不要提供与问题无关的信息，也不要重复。如果给定的上下文没有提供足够的信息，请说“关于相关主题的信息缺失”，然后是相关主题。

请使用引用编号格式引用上下文，格式为[citation:x]。如果一句话来自多个上下文，请列出所有适用的引用，例如[citation:3][citation:5]。除了代码和特定名称及引用外，你的回答必须使用与问题相同的语言。

以下是上下文集合：
{context}

记住，不要盲目地逐字重复上下文。以下是用户问题："""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

# This is the prompt that asks the model to generate related questions to the
# original question and the contexts.
# Ideally, one want to include both the original question and the answer from the
# model, but we are not doing that here: if we need to wait for the answer, then
# the generation of the related questions will usually have to start only after
# the whole answer is generated. This creates a noticeable delay in the response
# time. As a result, and as you will see in the code, we will be sending out two
# consecutive requests to the model: one for the answer, and one for the related
# questions. This is not ideal, but it is a good tradeoff between response time
# and quality.
_more_questions_prompt_en = """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""
_more_questions_prompt = """
你是一个有帮助的助手，根据用户的原始问题和相关上下文，帮助用户提出相关问题。请识别出值得进一步探讨的话题，并写出每个不超过20个字的问题。请确保在后续问题中包含具体内容，如事件、名称、地点等，以便它们可以独立提出。例如，如果原始问题询问“曼哈顿计划”，在后续问题中不要只说“该项目”，而要使用全名“曼哈顿计划”。你的相关问题必须与原始问题使用相同的语言。

以下是问题的上下文：
{context}

记住，根据原始问题和相关上下文，建议三个这样的进一步问题。不要重复原始问题。每个相关问题不应超过20个字。以下是原始问题：
"""

def search_with_bing(query: str, subscription_key: str):
    """
    Search with bing and return the contexts.
    """
    # logger.info(f"query bing with {query}")
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    # logger.info(f"query bing with response {response.status_code}")
    # logger.info(f"json response: {response.json()}")
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_google(query: str, subscription_key: str, cx: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_serper(query: str, subscription_key: str):
    """
    Search with serper and return the contexts.
    """
    payload = json.dumps({
        "q": query,
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    })
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content["knowledgeGraph"].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append({
                    "name": json_content["knowledgeGraph"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content["answerBox"].get("answer")
            if url and snippet:
                contexts.append({
                    "name": json_content["answerBox"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet","")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

def search_with_searchapi(query: str, subscription_key: str):
    """
    Search with SearchApi.io and return the contexts.
    """
    payload = {
        "q": query,
        "engine": "google",
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    }
    headers = {"Authorization": f"Bearer {subscription_key}", "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SEARCHAPI_SEARCH_ENDPOINT}"
    )
    response = requests.get(
        SEARCHAPI_SEARCH_ENDPOINT,
        headers=headers,
        params=payload,
        timeout=30,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []

        if json_content.get("answer_box"):
            if json_content["answer_box"].get("organic_result"):
                title = json_content["answer_box"].get("organic_result").get("title", "")
                url = json_content["answer_box"].get("organic_result").get("link", "")
            if json_content["answer_box"].get("type") == "population_graph":
                title = json_content["answer_box"].get("place", "")
                url = json_content["answer_box"].get("explore_more_link", "")

            title = json_content["answer_box"].get("title", "")
            url = json_content["answer_box"].get("link")
            snippet =  json_content["answer_box"].get("answer") or json_content["answer_box"].get("snippet")

            if url and snippet:
                contexts.append({
                    "name": title,
                    "url": url,
                    "snippet": snippet
                })

        if json_content.get("knowledge_graph"):
            if json_content["knowledge_graph"].get("source"):
                url = json_content["knowledge_graph"].get("source").get("link", "")

            url = json_content["knowledge_graph"].get("website", "")
            snippet = json_content["knowledge_graph"].get("description")

            if url and snippet:
                contexts.append({
                    "name": json_content["knowledge_graph"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })

        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic_results"]
        ]
        
        if json_content.get("related_questions"):
            for question in json_content["related_questions"]:
                if question.get("source"):
                    url = question.get("source").get("link", "")
                else:
                    url = ""  
                    
                snippet = question.get("answer", "")

                if url and snippet:
                    contexts.append({
                        "name": question.get("question", ""),
                        "url": url,
                        "snippet": snippet
                    })

        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

# app = FastAPI()

# 挂载静态文件目录
# app.mount("/ui", StaticFiles(directory="ui"), name="ui")


class SearchRequest(BaseModel):
    query: str
    search_uuid: str
    generate_related_questions: Optional[bool] = True,


class RAG(Photon):
    """
    Retrieval-Augmented Generation Demo from Lepton AI.

    This is a minimal example to show how to build a RAG engine with Lepton AI.
    It uses search engine to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions. The results
    are then stored in a KV so that it can be retrieved later.
    """

    requirement_dependency = [
        "openai",  # for openai client usage.
    ]

    extra_files = glob.glob("ui/**/*", recursive=True)

    deployment_template = {
        # All actual computations are carried out via remote apis, so
        # we will use a cpu.small instance which is already enough for most of
        # the work.
        "resource_shape": "cpu.small",
        # You most likely don't need to change this.
        "env": {
            # Choose the backend. Currently, we support BING and GOOGLE. For
            # simplicity, in this demo, if you specify the backend as LEPTON,
            # we will use the hosted serverless version of lepton search api
            # at https://search-api.lepton.run/ to do the search and RAG, which
            # runs the same code (slightly modified and might contain improvements)
            # as this demo.
            "BACKEND": "LEPTON",
            # If you are using google, specify the search cx.
            "GOOGLE_SEARCH_CX": "",
            # Specify the LLM model you are going to use.
            "LLM_MODEL": "mixtral-8x7b",
            # For all the search queries and results, we will use the Lepton KV to
            # store them so that we can retrieve them later. Specify the name of the
            # KV here.
            "KV_NAME": "search-with-lepton",
            # If set to true, will generate related questions. Otherwise, will not.
            "RELATED_QUESTIONS": "true",
            # On the lepton platform, allow web access when you are logged in.
            "LEPTON_ENABLE_AUTH_BY_COOKIE": "true",
        },
        # Secrets you need to have: search api subscription key, and lepton
        # workspace token to query lepton's llama models.
        "secret": [
            # If you use BING, you need to specify the subscription key. Otherwise
            # it is not needed.
            "BING_SEARCH_V7_SUBSCRIPTION_KEY",
            # If you use GOOGLE, you need to specify the search api key. Note that
            # you should also specify the cx in the env.
            "GOOGLE_SEARCH_API_KEY",
            # If you use Serper, you need to specify the search api key.
            "SERPER_SEARCH_API_KEY",
            # If you use SearchApi, you need to specify the search api key.
            "SEARCHAPI_API_KEY",
            # You need to specify the workspace token to query lepton's LLM models.
            "LEPTON_WORKSPACE_TOKEN",
        ],
    }

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    def local_client(self):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """
        import openai

        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                # base_url=f"https://{self.model}.lepton.run/api/v1/",
                base_url=os.environ.get("OPENAI_URL", OPENAI_URL),
                api_key=OPENAI_API_KEY, # os.environ.get("OPENAI_API_KEY")
                # or WorkspaceInfoLocalRecord.get_current_workspace_token(),
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client

    def init(self):
        """
        Initializes photon configs.
        """
        # First, log in to the workspace.
        # leptonai.api.workspace.login()
        self.backend = "BING" # os.environ["BACKEND"].upper()
        if self.backend == "LEPTON":
            self.leptonsearch_client = Client(
                "https://search-api.lepton.run/",
                token=os.environ.get("LEPTON_WORKSPACE_TOKEN")
                or WorkspaceInfoLocalRecord.get_current_workspace_token(),
                stream=True,
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
        elif self.backend == "BING":
            # self.search_api_key = BING_SEARCH_V7_SUBSCRIPTION_KEY
            self.search_api_key = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY", BING_SEARCH_V7_SUBSCRIPTION_KEY)
            self.search_function = lambda query: search_with_bing(
                query,
                self.search_api_key,
            )
        elif self.backend == "GOOGLE":
            self.search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
            self.search_function = lambda query: search_with_google(
                query,
                self.search_api_key,
                os.environ["GOOGLE_SEARCH_CX"],
            )
        elif self.backend == "SERPER":
            self.search_api_key = os.environ["SERPER_SEARCH_API_KEY"]
            self.search_function = lambda query: search_with_serper(
                query,
                self.search_api_key,
            )
        elif self.backend == "SEARCHAPI":
            self.search_api_key = os.environ["SEARCHAPI_API_KEY"]
            self.search_function = lambda query: search_with_searchapi(
                query,
                self.search_api_key,
            )
        else:
            raise RuntimeError("Backend must be LEPTON, BING, GOOGLE, SERPER or SEARCHAPI.")
        self.model = OPENAI_MODEL # os.environ["LLM_MODEL"]
        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        # Create the KV to store the search results.
        """
        logger.info("Creating KV. May take a while for the first time.")
        self.kv = KV(
            os.environ["KV_NAME"], create_if_not_exists=True, error_if_exists=False
        )
        """
        # whether we should generate related questions.
        self.should_do_related_questions = to_bool(os.environ["RELATED_QUESTIONS"])

    def get_related_questions(self, query, contexts):
        """
        Gets related questions based on the query and context.
        """

        def ask_related_questions(
            questions: Annotated[
                List[str],
                [(
                    "question",
                    Annotated[
                        str, "related question to the original question and context."
                    ],
                )],
            ]
        ):
            """
            ask further questions that are related to the input and output.
            """
            pass

        try:
            response = self.local_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _more_questions_prompt.format(
                            context="\n\n".join([c["snippet"] for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                tools=[{
                    "type": "function",
                    "function": tool.get_tools_spec(ask_related_questions),
                }],
                max_tokens=512,
            )
            related = response.choices[0].message.tool_calls[0].function.arguments
            if isinstance(related, str):
                related = json.loads(related)
            logger.trace(f"Related questions: {related}")
            return related["questions"][:5]
        except Exception as e:
            # For any exceptions, we will just return an empty list.
            logger.error(
                "encountered error while generating related questions:"
                f" {e}\n{traceback.format_exc()}"
            )
            return []

    def _raw_stream_response(
        self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            try:
                result = json.dumps(related_questions)
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

    def stream_and_upload_to_kv(
        self, contexts, llm_response, related_questions_future, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result and uploads to KV.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
            contexts, llm_response, related_questions_future
        ):
            all_yielded_results.append(result)
            yield result
        # Second, upload to KV. Note that if uploading to KV fails, we will silently
        # ignore it, because we don't want to affect the user experience.
        # _ = self.executor.submit(self.kv.put, search_uuid, "".join(all_yielded_results))
    
    @Photon.handler(method="POST", path="/bing/query")
    def query_bing(self, query: str) -> JSONResponse:
        """
        query bing
        """
        contexts = self.search_function(query)
        return JSONResponse(content=contexts)

    @Photon.handler(method="POST", path="/query")
    # @staticmethod
    # @app.post("/query")
    def query_function(
        self,
        query: str,
        search_uuid: str,
        generate_related_questions: Optional[bool] = True,
    ) -> StreamingResponse:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - search_uuid: a uuid that is used to store or retrieve the search result. If
                the uuid does not exist, generate and write to the kv. If the kv
                fails, we generate regardless, in favor of availability. If the uuid
                exists, return the stored result.
            - generate_related_questions: if set to false, will not generate related
                questions. Otherwise, will depend on the environment variable
                RELATED_QUESTIONS. Default: true.
        """
        # Note that, if uuid exists, we don't check if the stored query is the same
        # as the current query, and simply return the stored result. This is to enable
        # the user to share a searched link to others and have others see the same result.
        
        """
        if search_uuid:
            try:
                result = self.kv.get(search_uuid)

                def str_to_generator(result: str) -> Generator[str, None, None]:
                    yield result

                return StreamingResponse(str_to_generator(result))
            except KeyError:
                logger.info(f"Key {search_uuid} not found, will generate again.")
            except Exception as e:
                logger.error(
                    f"KV error: {e}\n{traceback.format_exc()}, will generate again."
                )
        else:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")
        """
        if self.backend == "LEPTON":
            # delegate to the lepton search api.
            result = self.leptonsearch_client.query(
                query=query,
                search_uuid=search_uuid,
                generate_related_questions=generate_related_questions,
            )
            return StreamingResponse(content=result, media_type="text/html")

        # First, do a search query.
        query = query or _default_query
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
            )
        )
        try:
            client = self.local_client()
            llm_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
            if self.should_do_related_questions: # and generate_related_questions:
                # While the answer is being generated, we can start generating
                # related questions as a future.
                related_questions_future = self.executor.submit(
                    self.get_related_questions, query, contexts
                )
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.stream_and_upload_to_kv(
                contexts, llm_response, related_questions_future, search_uuid
            ),
            media_type="text/html",
        )

    @Photon.handler(mount=True)
    # @app.get("/ui")
    def ui(self):
        return StaticFiles(directory="ui", html=True)

    @Photon.handler(method="GET", path="/")
    # @staticmethod
    # @app.get("/")
    def index(self) -> RedirectResponse:
        """
        Redirects "/" to the ui page.
        """
        return RedirectResponse(url="/ui/index.html")


if __name__ == "__main__":
    rag = RAG()
    rag.launch(port=BACKEND_PORT, log_level="debug")
# 启动应用
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
    