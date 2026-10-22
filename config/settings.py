"""
settings.py 配置管理模块 —— 事用Pydantic Settings 实现类型安全的配置。

.env文件只是存储配置的原材料，它本身不会被程序自动识别的，必须有代码显式的读取他。

Pydantic Settings 是一个用于管理应用程序配置的工具，它可以从多种来源加载配置，包括环境变量、命令行参数、.env文件等。
它还可以自动验证配置的类型和格式，并提供默认值和环境变量名称的映射。

最简单的理解，不设置的话，你需要再所有需要加载环境变量的文件中写代码，一处修改处处修改，很麻烦。
"""
from pydantic_settings import *
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import model_validator
from sqlalchemy.engine import URL

class Settings(BaseSettings):
    """
    配置管理类
    """
    # 项目基本配置
    PROJECT_NAME: str = "企业级RAG智能问答系统"
    PROJECT_VERSION: str = "1.0.0"
    BASE_DIR : Path = Path(__file__).parent.parent.resolve()

    # 数据目录
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_DB_DIR : Path = BASE_DIR / "vector_db"
    MODELS_DIR : Path = BASE_DIR / "models"
    UPLOAD_DIR : Path = BASE_DIR / "upload"

    # RAG闭环第一阶段：业务数据与可观测性
    # 配置本机MySQL账号后自动组装连接；未配置时兼容旧SQLite开发测试。
    DATABASE_URL : str = ""
    MYSQL_HOST : str = "127.0.0.1"
    MYSQL_PORT : int = 3306
    MYSQL_USER : str = ""
    MYSQL_PASSWORD : str = ""
    MYSQL_DATABASE : str = "rag_qa_system"
    SALES_QUERY_ENABLED : bool = True
    SALES_QUERY_TIMEOUT_SECONDS : float = 3
    SALES_QUERY_MAX_ROWS : int = 100
    SALES_SCHEMA_INDEX_DIR : Path = DATA_DIR / "sales_schema_index"
    SALES_SCHEMA_MIN_SIMILARITY : float = 0.35
    COMMERCE_DEMO_DATABASE_PATH : Path = DATA_DIR / "commerce_demo.sqlite3"
    COMMERCE_DEMO_MAX_PERIOD_DAYS : int = 120
    COMMERCE_ASSISTANT_ENABLED : bool = True
    LANGFUSE_ENABLED : bool = False
    LANGFUSE_PUBLIC_KEY : str = ""
    LANGFUSE_SECRET_KEY : str = ""
    LANGFUSE_HOST : str = "https://cloud.langfuse.com"
    LANGFUSE_CAPTURE_CONTENT : bool = False
    LANGFUSE_MAX_CONTENT_LENGTH : int = 4000

    # RAG阶段2/3：认证、任务、索引发布和质量门禁
    AUTH_REQUIRED : bool = False
    AUTH_SECRET_KEY : str = "change-me-in-production"
    AUTH_TOKEN_EXPIRE_MINUTES : int = 480
    AUTH_DEV_MODE : bool = True
    INGESTION_MAX_RETRIES : int = 3
    INGESTION_WORKERS : int = 2
    KNOWLEDGE_REVIEW_REQUIRED : bool = True
    TASK_EXECUTION_MODE : Literal["inline", "worker"] = "inline"
    TASK_LEASE_SECONDS : int = 120
    TASK_MAX_ATTEMPTS : int = 3
    TASK_POLL_SECONDS : float = 2
    MAX_UPLOAD_MB : int = 30
    MAX_ARCHIVE_EXPANDED_MB : int = 100
    MAX_ARCHIVE_ENTRIES : int = 2000
    MAX_PDF_PAGES : int = 500
    OCR_ENABLED : bool = True
    # 质量测试是否调用真实 OCR；只有本机明确开启时才会产生外部识别调用。
    PARSING_QUALITY_RUN_OCR : bool = False
    OCR_PROVIDER : Literal["tencent", "tesseract"] = "tencent"
    OCR_LANGUAGE : str = "chi_sim+eng"
    OCR_DPI : int = 150
    OCR_MIN_TEXT_CHARS : int = 20
    TENCENT_OCR_SECRET_ID : str = ""
    TENCENT_OCR_SECRET_KEY : str = ""
    TENCENT_OCR_REGION : str = "ap-guangzhou"
    TENCENT_OCR_ENDPOINT : str = "https://ocr.tencentcloudapi.com"
    TENCENT_OCR_ACTION : str = "GeneralBasicOCR"
    TENCENT_OCR_VERSION : str = "2018-11-19"
    TENCENT_OCR_TIMEOUT_SECONDS : float = 15
    DOCUMENT_CLEANING_ENABLED : bool = True
    REMOVE_REPEATED_PAGE_EDGES : bool = False
    CLARIFICATION_ENABLED : bool = True
    # 默认由大模型判断所有业务；False 仅用于显式切回旧关键词模式，不是模型失败兜底。
    INTENT_ROUTER_ENABLED : bool = True
    # Unified model/tool loop. False explicitly restores the legacy routing path.
    AGENT_ENABLED : bool = True
    AGENT_MAX_ROUNDS : int = 16
    AGENT_MAX_TOOL_CALLS : int = 24
    AGENT_TIMEOUT_SECONDS : float = 600
    # 模型自报低分时先追问；该分数不代表经过测试的真实正确率。
    INTENT_CONFIDENCE_THRESHOLD : float = 0.65
    INDEX_SNAPSHOT_DIR : Path = BASE_DIR / "data" / "index_snapshots"
    REQUIRE_EVIDENCE : bool = True
    EVIDENCE_MIN_SOURCES : int = 1
    EVAL_MIN_FAITHFULNESS : float = 0.6
    EVAL_MIN_ANSWER_RELEVANCY : float = 0.6
    EVAL_MIN_CONTEXT_PRECISION : float = 0.5
    EVAL_MIN_CONTEXT_RECALL : float = 0.5
    EVAL_MIN_ANSWER_CORRECTNESS : float = 0.6

    # 向量库配置
    VECTOR_STORE_TYPE : Literal["faiss", "chroma", "opensearch"] = "chroma"
    OPENSEARCH_URL : str = "https://localhost:9200"
    OPENSEARCH_USERNAME : str = "admin"
    OPENSEARCH_PASSWORD : str = ""
    OPENSEARCH_VERIFY_CERTS : bool = True
    OPENSEARCH_CA_CERTS : str | None = None
    OPENSEARCH_INDEX : str = "rag-chunks"
    OPENSEARCH_DIMENSION : int = 512
    OPENSEARCH_TIMEOUT : int = 30
    SEARCH_MIN_VECTOR_SIMILARITY : float = 0.35
    SEARCH_MIN_BM25_SCORE : float = 1.0
    RERANK_MIN_SCORE : float | None = None
    QUALITY_GUARD_ENABLED : bool = False
    ALLOW_GENERAL_ANSWERS : bool = True
    OTEL_ENABLED : bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT : str = "http://localhost:4318/v1/traces"
    OTEL_SERVICE_NAME : str = "rag-api"
    LLM_INPUT_COST_PER_MILLION : float = 0.0
    LLM_OUTPUT_COST_PER_MILLION : float = 0.0
    ALERT_RECEIVER_TOKEN : str = ""
    ALERT_NOTIFICATION_URL : str = ""
    S3_ENDPOINT_URL : str | None = None
    S3_ACCESS_KEY : str = ""
    S3_SECRET_KEY : str = ""
    S3_BUCKET : str = "rag-backups"
    S3_REGION : str = "us-east-1"
    EMBEDDING_MODEL_NAME : str = "bge_small_zh"
    EMBEDDING_DEVICE : str = "cpu"      # 可以配置cuda
    CHUNK_SIZE : int = 500
    CHUNK_OVERLAP : int = 50

    # 检索配置
    SEARCH_TOP_K : int = 5
    USE_RERANKER : bool = True
    RERANKER_MODEL_NAME : str = "bge_reranker_base"

    # LLM配置
    LLM_PROVIDER : Literal["openai", "siliconflow", "ollama"] = "openai"

    # ollama配置
    OLLAMA_BASE_URL : str = "http://localhost:11434"
    OLLAMA_MODEL_NAME : str = "qwen2.5:7b"

    # OpenAI配置
    OPENAI_API_KEY : str = ""
    OPENAI_BASE_URL : str = "https://api.tangguo.xin/v1"
    OPENAI_MODEL_NAME : str = "gpt-6.1-sol"

    # SiliconFlow配置
    SILICONFLOW_API_KEY : str = ""
    SILICONFLOW_BASE_URL : str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_MODEL_NAME : str = "Qwen/Qwen2.5-7B-Instruct"

    # 服务配置
    API_HOST : str = "localhost"
    API_PORT : int = 8000
    STREAMLIT_PORT : int = 8501

    # 日志配置
    LOG_LEVEL : str = "INFO"
    LOG_FILE : Path = BASE_DIR / "app.log"

    # 安全配置
    # ALLOWED_ORIGINS : list[str] = ["*"]       # * 是允许所有的
    ALLOWED_ORIGINS : list[str] = ["http://localhost:8501"]
    SECRET_KEY : str = "your-secret-key"

    @model_validator(mode="after")
    def resolve_database_url(self):
        if not self.DATABASE_URL:
            if self.MYSQL_USER:
                self.DATABASE_URL = URL.create("mysql+pymysql", username=self.MYSQL_USER,
                    password=self.MYSQL_PASSWORD, host=self.MYSQL_HOST, port=self.MYSQL_PORT,
                    database=self.MYSQL_DATABASE, query={"charset": "utf8mb4"}).render_as_string(hide_password=False)
            else:
                self.DATABASE_URL = f"sqlite:///{self.BASE_DIR / 'rag_qa.db'}"
        return self

    class Config:
        # 固定读取项目根目录的配置，避免 IDE 从 tests 启动时漏读 OCR 开关和凭据。
        env_file = Path(__file__).resolve().parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 实例化全局配置
settings = Settings()

# 确保必要目录存在
for dir_path in [
    settings.DATA_DIR,
    settings.VECTOR_DB_DIR,
    settings.MODELS_DIR,
    settings.UPLOAD_DIR,
    settings.INDEX_SNAPSHOT_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)
