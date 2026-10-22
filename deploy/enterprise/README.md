# 第3、10、12步企业部署

本目录提供 PostgreSQL、OpenSearch(BM25+CJK分词+Lucene向量)、MinIO、OpenTelemetry Collector、Tempo、Prometheus、Grafana、Alertmanager 和应用的单机部署配置。不是高可用集群，不代表已通过生产验收。

## 启动前

- 配置 `deploy/enterprise/.env`，字段参考 `.env.example`。不要修改项目原有 `.env`。
- 配置现有LLM地址、模型名、密钥，确保模型支持结构化输出，否则严格审核会拒答。
- `models` 目录须包含 `bge_small_zh` 和 `bge_reranker_base`；向量维数须与模型一致。
- Docker需运行Linux容器，至少预留8GB内存；OpenSearch需合适的 `vm.max_map_count`。
- `AUTH_DEV_MODE=false`，需由可信系统签发项目现有HS256令牌。未新增企业SSO，禁止在生产开启开发令牌。
- 本示例仅绑定本机端口，OpenSearch使用隔离网络中的演示证书且暂不校验证书。正式环境必须替换证书、启用CA验证、专用最小权限账户及TLS入口；不能直接暴露公网。
- 镜像固定到兼容版本以便重复验证，上线前必须自行做漏洞扫描和版本更新验证。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_enterprise.py
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml config --quiet
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml up --build -d
```

访问：API `http://localhost:8001/docs`；前端 `http://localhost:8502`；Grafana `http://localhost:3000`；Prometheus `http://localhost:9090`；Alertmanager `http://localhost:9093`。前端各业务页填写可信访问令牌，管理员使用“反馈处理”处理人工申请并转回归题。

## 扫描件 OCR

扫描 PDF 默认使用腾讯 OCR。复制 `.env.example` 为 `.env` 后设置 `TENCENT_OCR_SECRET_ID` 和 `TENCENT_OCR_SECRET_KEY`；页面图片会发送到腾讯云，可能产生调用费用。需要本机识别时设 `OCR_PROVIDER=tesseract`，镜像已包含 Tesseract 与中英文语言包。API 和后台 worker 共用此配置。

## 检索与审核

企业模式将关键词和向量存在同一个OpenSearch索引，两个查询都在Top-K之前带知识库授权过滤；余弦阈值和BM25阈值先筛选，RRF合并去重，再重排。阈值必须用实际测试集校准，不可把默认数值当质量承诺。

严格模式不依靠简单意图规则放行无证据回答；无证据直接拒答，有证据先生成带编号引用的回答，再独立检查安全、证据充分性和引用支持。审核失败或模型调用失败均不放行。审核模型并非事实保证，仍需Ragas与人工抽样。每次审核会增加一次LLM调用和延迟。

监控只导出计数、耗时、Token和估算成本，不把问题正文或用户/文档ID作为指标标签。必须按所用模型设置Token单价，否则成本为0。告警经独立令牌验证后入库；可设置 `ALERT_NOTIFICATION_URL` 转发到通知网关。邮箱由 Alertmanager 直接发送，钉钉由独立的 `alert-notifier` 发送，不依赖业务 API 正常运行。通知默认关闭，启用字段见 `.env.example`；配置后运行 `scripts/prepare_enterprise.py --mode docker --force` 并重建告警组件。本地宿主机模式使用 `--mode local`。完整步骤见 [监控与链路追踪运维手册](../../监控与链路追踪运维手册.md)。默认阈值在 `alerts.yml`，需按业务基线调整。

## 历史数据

新容器使用独立数据卷，不自动迁移现有SQLite。先备份，把SQLite/上传目录/向量库的复制品只读挂载到维护容器。API启动前，在全新空PostgreSQL运行 `scripts/migrate_business_data.py /legacy/rag_qa.db --source-upload-root 原上传根目录` 预览，再加 `--execute` 导入；记录ID保留，文件路径映射到容器目录。把上传文件复制到新上传卷并校验后，运行 `scripts/migrate_opensearch.py --source chroma --source-dir /legacy/vector_db --index rag-chunks --source-upload-root 原上传根目录`。迁移向量前先初始化目标数据库表。FAISS迁移只允许可信备份，读取其文件涉及pickle反序列化。不要先把线上流量切到空库。

现有本地文件索引快照不能用于OpenSearch，相关入口会明确拒绝。候选索引、审核、完整评测、灰度和回退已接入管理页面，操作顺序见 [第1–5批交付与验收](../../批次1-5交付与验收.md)。模板包含独立worker；与API共用数据库、上传目录和模型配置。

## 备份和恢复

先停止API与其他入库写入方，再离线导出数据库、文件和OpenSearch向量，向量不会重新计算。包含敏感业务内容，限制备份目录和桶权限；对象存储需配置TLS、加密、异地副本与保留周期。

```powershell
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml stop backend frontend worker
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml run --rm maintenance python scripts/enterprise_backup.py create /backups/first.backup.zip --confirm-offline --upload
docker compose --env-file deploy/enterprise/.env -f deploy/enterprise/compose.yml start backend frontend worker
```

创建独立恢复环境，使用**全新空数据库、空OpenSearch索引和空文件卷**，容器路径/模型/维数与原环境一致。先运行 `restore ... --confirm-offline` 校验，再加 `--execute` 真正恢复。工具拒绝覆盖现有数据；跨组件恢复无法保证一个事务，失败后保留原环境并在新的空环境重新恢复。恢复完成后验证数据库数量、搜索数量、权限、反馈和实际问答，才允许切流量。

S3下载使用 `download /backups/recovery.backup.zip --key backups/first.backup.zip`；会核对对象校验和。备份频率和恢复演练由运维调度，此次没有自动停生产服务或自动覆盖数据。
