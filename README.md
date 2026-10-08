# 企业级私有化 RAG 智能问答系统

## 📖 项目简介

**企业级私有化 RAG 智能问答系统** 是一个面向中小企业/团队的内部知识库问答平台。支持上传 PDF、Word、Excel、TXT 等文档，自动向量化存储，并通过大语言模型生成精准回答。系统完全私有化部署，保障数据安全，适合客服、研发、行政等多种场景。

### ✨ 核心功能

- 📂 **多格式文档支持**：PDF、Word、Excel、TXT、Markdown
- 🧠 **智能问答**：基于检索增强生成（RAG），回答准确且可溯源
- 💬 **多轮对话**：支持上下文连续对话记忆
- ⚙️ **灵活部署**：支持 Ollama 本地模型或云端 API（硅基流动/OpenAI）
- 🌐 **可视化界面**：Streamlit 搭建的友好 Web 前端
- 🔧 **模块化设计**：清晰的分层架构，易于二次开发

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | Streamlit |
| **后端** | FastAPI |
| **大模型** | Qwen2.5-7B (Ollama/硅基流动) |
| **嵌入模型** | BAAI/bge-small-zh-v1.5 |
| **向量数据库** | Chroma / FAISS |
| **文档处理** | PyMuPDF、python-docx、Unstructured |
| **框架** | LangChain (LCEL) |
| **部署** | Docker + Docker Compose |

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- [Ollama](https://ollama.com/)（可选，如需本地模型）
- Git

### 2. 克隆项目

```bash
git clone https://github.com/sunwy515/rag-qa-system.git
cd rag-qa-system
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置文件

复制 `.env.example` 为 `.env`，并填写您的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# LLM配置示例（使用硅基流动）
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct

# 或使用本地Ollama
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b
```

### 5. 启动后端 API

```bash
python run_api.py
```

API 文档将自动生成在：http://localhost:8000/docs

### 6. 启动前端界面（新终端）

```bash
python run_frontend.py
```

浏览器访问：http://localhost:8501

---

## 🐳 Docker 一键部署（可选）

```bash
cd docker
docker-compose up -d
```

服务将自动启动：
- API：http://localhost:8000
- 前端：http://localhost:8501

---

## 📁 项目结构

```
rag-qa-system/
├── config/                 # 配置管理（Pydantic Settings）
│   ├── settings.py         # 所有配置项集中管理
│   └── logging_config.py   # 日志配置
├── core/                   # 核心业务逻辑（重点讲解）
│   ├── document_loader.py  # 文档加载与分块
│   ├── embedding.py        # 嵌入模型加载（单例）
│   ├── vector_store.py     # 向量库增删查（Chroma/FAISS）
│   ├── llm_client.py       # 大模型客户端（工厂模式）
│   ├── retriever.py        # 检索器（支持重排序）
│   ├── memory_manager.py   # 多轮对话记忆管理
│   └── rag_chain.py        # LCEL 完整问答链
├── api/                    # FastAPI 后端
│   ├── routes/
│   │   ├── upload.py       # 文件上传接口
│   │   ├── qa.py           # 问答接口
│   │   ├── knowledge.py    # 知识库管理
│   │   └── system.py       # 系统状态
│   └── main.py             # 应用入口
├── frontend/               # Streamlit 前端
│   ├── app.py              # 首页
│   ├── pages/
│   │   ├── 1_💬_智能问答.py
│   │   ├── 2_📁_知识库管理.py
│   │   └── 3_⚙️_系统设置.py
│   └── components/         # 可复用组件
├── utils/                  # 工具函数
├── data/                   # 上传文件存储（自动创建）
├── vector_db/              # 向量库持久化目录
├── tests/                  # 单元测试脚本
├── requirements.txt        # 依赖清单
├── run_api.py              # 后端启动脚本
└── run_frontend.py         # 前端启动脚本
```

---


## 📡 API 接口速览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/system/health` | 健康检查 |
| POST | `/api/v1/upload/file` | 上传单个文件 |
| POST | `/api/v1/qa/ask` | 问答请求 |
| POST | `/api/v1/qa/clear_memory` | 清除对话记忆 |
| GET | `/api/v1/knowledge/stats` | 知识库统计 |
| POST | `/api/v1/knowledge/clear` | 清空知识库 |

详细文档访问：http://localhost:8000/docs

---

## 🔍 常见问题

### Q1：启动后端报错 `ImportError: cannot import name 'BaseDocumentCompressor'`
**A**：修改 `core/retriever.py`，将导入改为：
```python
from langchain_core.documents.compressor import BaseDocumentCompressor
```

### Q2：Ollama 连接失败
**A**：确保 Ollama 服务已启动，并已拉取模型：
```bash
ollama serve
ollama pull qwen2.5:7b
```

### Q3：上传文件后问答无结果
**A**：检查 `uploads/` 目录权限，确认向量库是否正常写入。可运行模块3测试验证。

### Q4：前端显示“API离线”
**A**：确保后端服务运行在 `http://localhost:8000`，且防火墙未拦截。

---


## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request！  
如果您在使用过程中有任何疑问，也欢迎在课程讨论区提问。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。