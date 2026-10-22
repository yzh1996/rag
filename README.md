# 企业级私有化 RAG 智能问答系统

## 1. 项目定位

本项目是一个可私有化部署的企业知识问答与数据分析平台。默认由统一 Agent 结合当前问题与已确认任务信息，决定直接回答、检索内部知识、只读查询业务数据、计算比较或追问；程序执行权限与参数检查，记录每步结果。线上反馈可以进入测试集、评测和索引发布流程。

当前项目同时提供两条数据查询能力：

- **企业知识问答**：从已发布知识库检索 PDF、Word、Excel、TXT、Markdown 等资料，支持多轮对话、引用、澄清和拒答。
- **销量与经营分析**：本机 MySQL 模式按 `doc` 字段字典查询 `shop`、`jl_user`、`live_base`、`live_sales_info`、`live_product_sale`、`product` 等业务表；模拟记录统一在 MySQL 中生成。模型按需组合查数、知识检索和计算，支持店铺、主播、直播和商品问题。

项目包含候选索引、人工审核、离线评测、质量门禁、灰度发布、回滚、持久化任务、OCR 和运行监控等企业化能力。企业部署模板已经提供，但真实企业登录、生产数据迁移、高可用和正式上线仍需单独验收。

## 2. 当前能力

- 文档上传前安全检查、重复文件检查、公司/租户和部门归属控制。
- 上传后预览解析结果和分块结果，确认后才进入向量化入库。
- PDF、Word、Excel、TXT、Markdown 解析；扫描 PDF 支持腾讯 OCR 或本机 Tesseract。
- 递归、固定长度、结构化、语义和父子分块策略。
- 文档版本、权限范围、下线和恢复。
- 向量检索、关键词检索、权限前置过滤、合并去重、可选重排和父段落补回。
- 多轮会话、流式问答、引用来源、证据不足时澄清或拒答。
- 用户反馈、人工处理、正确答案维护和回归题沉淀。
- RAG 测试集、后台评测、Ragas 指标、候选索引绑定和发布质量门禁。
- 候选索引审核、按租户灰度、稳定分流和已发布版本回滚。
- 持久化后台任务、任务去重、租约、自动重试、过期恢复和独立 Worker。
- 按业务字段字典核对 MySQL 表结构的只读 Text-to-SQL 查询，包含 SQL 检查、查询超时和结果上限。
- 同一 MySQL 库中的 90 天模拟数据；统一工具循环有次数、耗时和重复失败上限，页面展示实际执行过程。
- 已确认对象、场次和查询依据随问答持久化；下一轮恢复并检查身份、权限配置、字典版本和有效期。
- Prometheus 指标、OpenTelemetry/Tempo 链路、Grafana 看板、Alertmanager 告警和可选 Langfuse 追踪。

## 3. 技术架构

```text
Streamlit 前端
    ↓ HTTP / SSE
FastAPI 接口层
    ↓ 认证、租户隔离、参数校验、路由分发
core 核心服务
    ├─ RAGChain/ConversationAgent：恢复任务、模型决策、工具执行、回答与保存
    ├─ Upload/Ingestion：上传、清洗、分块、入库任务
    ├─ Retriever：关键词+向量检索、权限过滤、重排
    ├─ Evaluation/Release：评测、质量门禁、索引发布和回滚
    ├─ AgentTools：字段说明、只读 SQL、知识检索、结果计算
    └─ TaskQueue/Telemetry：任务可靠性和可观测性
    ↓
业务数据库 / 向量检索 / 文件存储 / Embedding / LLM / OCR
```

### 本地模式与企业模式

| 项目 | 本地默认模式 | 企业部署模板 |
| --- | --- | --- |
| 业务数据库 | 本机 MySQL；未配置时兼容 SQLite | PostgreSQL |
| 普通知识检索 | Chroma，亦支持 FAISS | OpenSearch（BM25+CJK+向量） |
| 文件和备份 | `upload/`、`data/` | Docker 卷、MinIO、备份卷 |
| 任务执行 | API 内置本地线程 | 独立 `worker` 容器 |
| 登录 | `AUTH_REQUIRED=false`，开发模式 | `AUTH_REQUIRED=true`、`AUTH_DEV_MODE=false` |
| 质量审核 | 默认关闭 | `QUALITY_GUARD_ENABLED=true` |
| 无证据的通用回答 | 默认允许 | `ALLOW_GENERAL_ANSWERS=false` |
| 链路追踪 | 默认关闭 | OpenTelemetry Collector + Tempo |

企业模板是单机部署参考，不等于高可用生产集群。端口、证书、密钥、身份系统和容量需要按实际环境重新验收。

## 4. 环境准备

- Windows 或 Linux，Python 3.10+。
- 本地开发需要可用的 LLM 服务：Ollama、OpenAI 兼容接口或 SiliconFlow。
- 使用本地 Embedding/重排时，`models/` 中应准备 `bge_small_zh` 和 `bge_reranker_base` 对应模型。
- 扫描 PDF 使用腾讯 OCR 时，需要配置腾讯云密钥；使用 Tesseract 时，需要安装程序及 `chi_sim`、`eng` 语言包。
- 企业模式需要 Docker Compose、至少约 8GB 可用内存，并按 OpenSearch 要求设置主机参数。

## 5. 本地启动

项目默认从根目录 `.env` 和环境变量读取配置。先按 [本机MySQL业务数据说明](doc/mysql_business_data.md) 填写 `MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE` 并初始化。不要把真实密钥提交到 Git。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/seed_mysql_business.py --create-database
.\.venv\Scripts\python.exe run_api.py
```

另开终端启动前端：

```powershell
.\.venv\Scripts\python.exe run_frontend.py
```

默认访问地址：

- API 文档：`http://localhost:8000/docs`
- Streamlit：`http://localhost:8501`
- 健康检查：`http://localhost:8000/api/v1/system/health`
- Prometheus 指标：`http://localhost:8000/api/v1/system/metrics`

配置 MySQL 后使用本机 MySQL、Chroma、内置任务线程和开发身份；未配置 MySQL 时兼容旧 SQLite。多人使用前应启用认证，不能把本地开发模式当作企业权限隔离。

### 独立 Worker 模式

将 `TASK_EXECUTION_MODE` 设置为 `worker` 后，API 只负责接收和登记任务，另开进程执行：

```powershell
.\.venv\Scripts\python.exe run_worker.py
```

API 和 Worker 必须使用同一个数据库、上传目录、模型目录和配置。否则任务状态、文件和索引会不一致。

### 本机 MySQL 模拟数据

以下命令按字典中的原表名造数，不覆盖已有业务记录，不会在服务启动时自动执行：

```powershell
.\.venv\Scripts\python.exe scripts/seed_mysql_business.py --create-database
```

64 张表中 49 张可建表、47 张生成模拟记录；15 张缺失字段清单、2 张关键定义不足，详见 [MySQL业务数据说明](doc/mysql_business_data.md)。问答页输入：**查询各店铺的直播销售额排名。**

默认启用 `AGENT_ENABLED=true`，不再先做互斥意图路由。可以提问“复盘清研美妆旗舰店最新一场直播”，再追问“那和上一场比呢？”。模型自行决定查询顺序；已确认的场次可用于追问，不重新套用店铺专用流程。没有退款、利润或购买转化等字段时必须说明缺失。

Agent 先接收简短能力目录，需要时获取相关工具和表字段说明；`query_database` 使用现有 SQL 检查和 MySQL 只读事务，`calculate` 引用实际结果计算。任务保存在 `qa_interactions.query_details.task_state`，无需变更业务表。默认最多 16 轮、24 次工具调用、600 秒处理预算；分别通过 `AGENT_MAX_ROUNDS`、`AGENT_MAX_TOOL_CALLS`、`AGENT_TIMEOUT_SECONDS` 调整。历史回答和查询依据不会仅因时间经过而清除；询问当前或最新数据时仍需重新查询，权限或业务表结构变化时也不会复用旧依据。支持原生工具调用的模型优先使用原生协议，其他模型使用结构化 JSON 工具协议，失败时明确提示。

设置 `AGENT_ENABLED=false` 可显式恢复旧 RAG/销量/主播分析分流，以便回退；`INTENT_ROUTER_ENABLED` 仅影响该旧模式。历史店铺报告仍可展示。本地业务数据目前是 default 租户的共享模拟数据，尚无部门级数据隔离；其他租户不能通过 Agent 访问该共享库。知识检索继续使用现有权限和发布版本检查。

升级后重启后端会自动给系统会话表补充 `latest_interaction_id`，用于准确指向最新问答，避免 MySQL 秒级时间导致轮次混淆；业务表和已有问答数据保留。问答执行方式变化会使旧索引评测配置失效，发布新版本前需要重新评测。

接近轮次或工具预算时保留最后一次整理机会，停止新查询并用已有数据返回标注为“部分结果”的回答；模型超时、连续工具失败仍会明确显示失败。默认流式接口持续发送进度，Agent 正文在查询和必要审核完成后发送，复杂问题可能需要数分钟。

旧 `seed_sales_data.py` 和 `seed_commerce_demo.py` 仅用于兼容旧 SQLite 演示，不是 MySQL 初始化步骤。

## 6. 企业部署

企业部署文件位于 `deploy/enterprise/`。它提供 PostgreSQL、OpenSearch、MinIO、Backend、Frontend、Worker、OpenTelemetry、Tempo、Prometheus、Alertmanager、Grafana 和维护容器。

1. 复制 `deploy/enterprise/.env.example` 为 `deploy/enterprise/.env`，填写唯一密码、认证密钥、模型地址和需要的通知配置。
2. 确保 `models/` 中有与配置一致的 Embedding 和重排模型。
3. 生成告警接收配置并检查 Compose：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_enterprise.py
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml config --quiet
```

4. 启动服务：

```powershell
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml up --build -d
```

默认访问地址：

- API：`http://localhost:8001/docs`
- 前端：`http://localhost:8502`
- Grafana：`http://localhost:3000`
- Prometheus：`http://localhost:9090`
- Alertmanager：`http://localhost:9093`

部署前必须阅读 [`deploy/enterprise/README.md`](deploy/enterprise/README.md)。模板默认只绑定本机端口，使用演示证书配置，不代表已完成生产安全验收。正式环境需要接入可信身份、替换证书、启用 CA 校验、配置最小权限账号、限制网络暴露并完成备份恢复演练。

## 7. 前端页面

| 页面 | 用途 |
| --- | --- |
| `1_💬_问答.py` | 多轮问答、流式输出、引用、销量查询、反馈 |
| `2_📁_知识库管理.py` | 上传预览、分块确认、文档权限、版本与索引发布、后台任务 |
| `3_⚙️_系统设置.py` | 服务状态和运行配置查看 |
| `4_📊_RAG评测.py` | 测试集、评测运行、指标和逐题结果 |
| `5_反馈处理.py` | 管理员处理负反馈、人工申请和回归题 |

## 8. API 功能分组

所有接口前缀为 `/api/v1`，完整参数以 `/docs` 为准。

| 路由 | 用途 |
| --- | --- |
| `/auth` | 开发令牌 |
| `/qa` | 普通问答、流式问答、历史、会话、反馈 |
| `/upload` | 上传、预览、确认、批量上传、任务重试 |
| `/documents` | 文档列表、版本、权限、下线和恢复 |
| `/knowledge` | 知识库统计、删除和清空 |
| `/jobs` | 入库和后台任务查询、重试 |
| `/evaluation` | 测试集导入、评测运行和结果 |
| `/index` | 候选索引、审核、发布、灰度和回滚 |
| `/retrieval` | 独立检索接口 |
| `/feedback` | 反馈列表、处理和转回归题 |
| `/operations` | 告警接收和管理员查询 |
| `/system` | 健康检查、指标和非敏感配置 |

## 9. 主要数据对象

主业务数据库包含问答、销量、反馈、评测、用户、知识库、文档版本、入库任务、索引发布和告警等表。核心表包括：

```text
qa_interactions       问答记录、引用、模型、耗时和追踪ID
qa_feedback           用户反馈、人工结论和处理状态
shop / jl_user        店铺与主播
live_base             直播场次
live_sales_info       直播销售额
live_product_sale     直播商品明细
product               商品
date_product_video_incr 视频商品日销售
business_datasets     模拟数据来源、约定和数量
eval_cases            评测题和权限测试上下文
eval_runs/results     评测运行和逐题结果
users                 用户、角色、部门和租户
knowledge_bases       知识分组和上传范围
documents/versions    文档、内容版本、状态和权限
ingestion_jobs        文档入库任务
upload_tasks          上传预览和确认任务
index_versions        候选索引与发布版本
release_states        每个租户的正式、灰度和上一版本指针
background_jobs       持久化任务队列
operational_alerts    告警接收记录
```

MySQL 经营数据按 `config/business_schema.json` 建立，该文件从 Word 字段字典提取。旧 `rag_qa.db` 和 `data/commerce_demo.sqlite3` 保留，不自动迁移或覆盖。

## 10. 文档与流程入口

- [项目主要流程解析](项目主要流程解析.md)：按请求链路解释模块关系、数据流、任务、权限、评测和发布。
- [RAG 闭环实施](RAG闭环实施.md)：建设原则、阶段目标和质量门槛。
- [第 1–5 批交付与验收](批次1-5交付与验收.md)：当前已实现能力和验收边界。
- [上传到分块流程交付](上传流程交付与验收.md)：上传、清洗、分块和恢复规则。
- [Text-to-SQL 接入说明](Text-to-SQL接入说明.md)：销量表、SQL 安全边界和模拟数据。
- [监控与链路追踪运维手册](监控与链路追踪运维手册.md)：指标、链路、告警和容量管理。
- [本机MySQL业务数据](doc/mysql_business_data.md)：字段字典、造数、连接配置与经营查询。

## 11. 验证

隔离测试使用临时数据库、临时向量库和模拟模型，不应读写当前业务数据库，也不应调用付费模型。可运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests
```

真实上线前还需要使用实际文件、实际模型、真实 OCR、PostgreSQL/OpenSearch 和企业身份系统完成独立验收。测试通过不等于生产质量、容量或安全性已经得到保证。

## 12. 已知边界

- 默认本地认证关闭，不适合多人生产使用。
- 本机 MySQL/Chroma 支持本地业务模拟；多进程企业部署应使用 PostgreSQL/OpenSearch，并让 API 与 Worker 共享存储。旧 SQLite 兼容模式只适合隔离开发测试。
- 旧数据迁移不会由 Compose 自动完成，必须先备份并使用迁移脚本在空环境预览和验证。
- OCR 会受扫描质量、表格和复杂版式影响，必须人工抽检。
- RAG 评测、质量门禁和模型审核不能替代业务人员抽检。
- 本机 MySQL 具备模拟销售额和每日明细，但仍缺少退款、成本及去重订单，不据此推算利润或购买转化率。
- MySQL 经营记录均为模拟数据；真实库的字段、金额单位和关联关系仍需核实。向量库保留 Chroma，不迁入 MySQL。

## 13. 许可证

项目当前保留原有许可证声明；正式发布前请根据实际仓库授权文件再次确认。
