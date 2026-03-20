import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma

#------------------------检索器构建-------------------------
def get_retriever():
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", base_url="https://jeniya.top/v1")
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

#------------------------提取context合并------------------
def combine_docs(docs):
    formatted_docs = []
    for doc in docs["context"]:
        # 尝试获取来源信息，如果没有则显示“未知来源”
        source = doc.metadata.get("source", "未知来源") 
        content = doc.page_content
        # 将来源和内容打包在一起
        formatted_docs.append(f"【来源：{source}】\n{content}")
    
    return "\n\n".join(formatted_docs)

#------------------------获取问答链-------------------------
def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0, base_url="https://jeniya.top/v1") 
    #---------------------------补充，完善问题的模板-----------------------------------
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
    #---------------------------根据有无历史记录检索----------------------------------
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )
    #---------------------------定义问答模板--------------------------
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "检索到的上下文{context}"
    )
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    # ---------------------------副链问答链----------------------------
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    # ---------------------------主链----------------------------
    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain
#------------------------生成回复-------------------------
def gen_response(chain, input, chat_history):
    response = chain.stream({                   #流式输出
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        # 1. 拦截底层检索结果：当数据流中出现 context 时，我们把它存下来
        if "context" in res:
            # 这里的 res["context"] 是原生的 List[Document] 列表
            st.session_state.current_context = res["context"]
            
        # 2. 正常透传大模型的文字回答
        elif "answer" in res:
            yield res["answer"]

def main():
    _ = load_dotenv(find_dotenv())
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    #在网页最顶部，用markdown画一个三级标题
    st.markdown('### RAG检索回答AI')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    # 显示整个对话历史
    for message in st.session_state.messages: 
        # 1. 先明确解析出角色和内容，避免后续混淆
        role = message[0]
        content = message[1]
        
        # 2. 开启对应的头像气泡框
        with messages.chat_message(role): 
            st.write(content) 
            
            # 3. ⚠️ 注意这里的缩进！它必须在 with 块的内部，
            # 这样折叠面板才会和文字一起被包在 AI 的气泡框里。
            if role == "ai" and len(message) > 2 and message[2]:
                docs = message[2]
                with st.expander("📚 查看参考来源与检索内容"):
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get("source", "未知来源")
                        st.markdown(f"**[{i+1}] 来源:** `{source}`")
                        st.info(doc.page_content)

    if prompt := st.chat_input("Say something"):
        
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)

        # 生成回复
        clean_chat_history = [(msg[0], msg[1]) for msg in st.session_state.messages]
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=clean_chat_history
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
            current_docs = st.session_state.get("current_context", [])
            if current_docs:
                with st.expander("📚 查看参考来源与检索内容"):
                    for i, doc in enumerate(current_docs):
                        source = doc.metadata.get("source", "未知来源")
                        st.markdown(f"**[{i+1}] 来源:** `{source}`")
                        st.info(doc.page_content)
                        
            # 我们依然需要清理临时变量，防止干扰流式输出的判定逻辑
            if "current_context" in st.session_state:
                del st.session_state.current_context

        # 4. 【严谨的存储】：将 context 作为第三个元素存入历史记录
        st.session_state.messages.append(("human", prompt))
        # 即使 current_docs 是空列表，也作为第三个元素占位，保持数据结构一致性
        st.session_state.messages.append(("ai", output, current_docs))


if __name__ == "__main__":
    main()

