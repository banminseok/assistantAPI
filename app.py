import streamlit as st
from openai import OpenAI
import openai 
import os
import time
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
)

st.title("Research Assistant 🔍")

# 사이드바
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key")    
    st.markdown("---")
    st.markdown(
        "[View Code on GitHub](https://github.com/banminseok/assistantAPI)", 
        unsafe_allow_html=True
    )

# API 키 유효성 검사
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# OpenAI 클라이언트 초기화
if api_key:
    client = OpenAI(api_key=api_key)
    os.environ["OPENAI_API_KEY"] = api_key
else:
    client = None

st.write("Welcome! I can help you research topics using Wikipedia, DuckDuckGo, and Web Scraping.")

# -----------------
# 도구 함수
# -----------------

def wikipedia_search(inputs):
    # Wikipedia를 검색합니다.
    query = inputs["query"]
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

def duckduckgo_search(inputs):
    # DuckDuckGo를 검색합니다.
    query = inputs["query"]
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)

def get_web_content(inputs):
    # URL에서 콘텐츠를 스크래핑하고 추출합니다.
    url = inputs["url"]
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if documents:
            content = "\n\n".join([doc.page_content for doc in documents])
            # 토큰 제한을 피하기 위해 콘텐츠 제한
            if len(content) > 10000:
                content = content[:10000] + "\n\n... (Content truncated)"
            return f"Content from {url}:\n\n{content}"
        else:
            return f"No content found at {url}."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# 함수 실행을 위한 매핑
functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "get_web_content": get_web_content,
}

# OpenAI 도구 정의
functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Wikipedia.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search the web using DuckDuckGo to find relevant information or URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_web_content",
            "description": "Scrapes and extracts complete text content from a given URL.  Preserves all original formatting and structure. Returns the full page content without summarization or truncation. Ideal for gathering comprehensive information from web sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape.",
                    }
                },
                "required": ["url"],
            },
        }
    }
]

# -----------------
# 헬퍼 함수
# -----------------

def get_tool_outputs(run_id, thread_id):
    run = client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        
        # 인수 파싱
        try:
            args = json.loads(function.arguments)
        except json.JSONDecodeError:
            args = {} 
            
        # 도구 실행
        output_result = functions_map[function.name](args)
        
        outputs.append(
            {
                "output": str(output_result),
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

# -----------------
# 이벤트 핸들러
# -----------------

class EventHandler(openai.AssistantEventHandler):
    """
    Assistant 스트림을 위한 이벤트 핸들러입니다.
    """
    def __init__(self):
        super().__init__()
        self.message_box = None
        self.current_message = ""

    @openai.override
    def on_text_created(self, text) -> None:
        # 새 메시지를 위한 컨테이너 생성
        self.message_box = st.empty()

    @openai.override
    def on_text_delta(self, delta, snapshot):
        # 토큰 누적 및 UI 업데이트
        self.current_message += delta.value
        if self.message_box:
            self.message_box.markdown(self.current_message.replace("$", "\$"))

    @openai.override
    def on_event(self, event):
        # 도구 호출을 위한 'requires_action' 처리
        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)

# -----------------
# 메인
# -----------------

ASSISTANT_NAME = "Research Assistant Agent"

# Assistant 및 스레드 초기화
if "assistant" not in st.session_state:
    try:

        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions="""
            You are a research documentation agent.
            1. Search Wikipedia for basic information using 'wikipedia_search'.
            2. Search DuckDuckGo using 'duckduckgo_search' to find relevant website URLs.
            3. Use WebScraper to scrape at least one relevant URL to get detailed information.
            4. Compile all the information into a comprehensive answer.
            """,
            model="gpt-4o-mini",
            tools=functions,
        )
        
        thread = client.beta.threads.create()
        
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
        
    except Exception as e:
        st.error(f"Failed to initialize Assistant: {e}")
        st.stop()
else:
    assistant = st.session_state["assistant"]
    thread = st.session_state["thread"]


# 채팅 인터페이스

def paint_history():
    # 스레드에서 메시지 검색
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(list(messages)):
        role = msg.role
        content = msg.content[0].text.value
        with st.chat_message(role):
            st.markdown(content)

# 
try:
    paint_history()
except Exception as e:
    st.error(f"Error loading chat history: {e}")

# 채팅 입력
query = st.chat_input("What do you want to research?")

if query:
    # 1. 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(query)
    
    # 2. 스레드에 메시지 추가
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
    )
    
    # 3. 스트림 실행
    with st.chat_message("assistant"):
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()


