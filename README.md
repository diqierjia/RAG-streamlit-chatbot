# 🦜🔗 RAG Streamlit Chatbot

本项目是一个基于大语言模型（LLM）和检索增强生成（RAG）技术的交互式问答助手。项目前端基于 Streamlit 构建，底层检索与问答逻辑由 LangChain 和 Chroma 向量数据库驱动。

## ✨ 核心逻辑与特性

1. **状态解耦**：严格分离了前端 UI 渲染状态与底层的 LangChain 数据流转，防止历史记录在多轮对话中被重复喂给大模型。
2. **精准溯源**：在流式输出（Streaming）的过程中拦截底层检索数据，并在 AI 回复下方以折叠面板的形式，严谨展示生成该回答所依赖的本地向量库文献及来源路径。
3. **记忆重写**：包含一个意图重写链路（`condense_question_prompt`），能够结合历史聊天记录，将用户的多轮简短提问还原为完整的独立问题，再进行向量检索。

## 🛠️ 环境配置与依赖安装

请严格按照以下步骤在本地复现本项目：

**1. 克隆项目到本地**
```bash
git clone [https://github.com/diqierjia/RAG-streamlit-chatbot.git](https://github.com/diqierjia/RAG-streamlit-chatbot.git)
cd RAG-streamlit-chatbot
```
**2. 创建并激活独立的虚拟环境**

```bash
python -m venv .venv
# Windows 激活命令：
.\.venv\Scripts\activate
# macOS/Linux 激活命令：
# source .venv/bin/activate
```

**3. 安装依赖**

```Bash
pip install -r requirements.txt
```

## 🔐 密钥与环境变量配置

**⚠️ 注意：出于安全考虑，本项目的 API 密钥并未硬编码在代码中，也没有上传到云端。**
你需要在项目根目录下手动新建一个名为 .env 的文件，并在其中写入你的 OpenAI API 配置信息：
```代码块
OPENAI_API_KEY="sk-你的真实密钥请填在这里"
# 如果你使用的是代理地址，也可在此配置
```


## 🚀 启动与运行
确保虚拟环境已激活，且向量数据库文件夹 data_base 路径完整，在终端运行以下命令启动服务：

```Bash
streamlit run streamlit_app.py
运行后，程序会自动在你的默认浏览器中弹出一个 Web 交互界面。
```

---

你可以在本地保存这个文件后，在 VS Code 里按 `Ctrl + Shift + V` 预览一下排版效果。确认无误后，再次执行 `git add .`、`git commit -m "修复 README 排版错误"` 和 `git push`。

