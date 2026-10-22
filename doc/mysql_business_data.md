# 本机 MySQL 业务数据与经营查询

业务数据范围以 `六个维度大数据平台数据表与字段字典.docx` 为准。`config/business_schema.json` 逐项保留原文表名、字段、类型、可空性、含义和来源；可用 `scripts/extract_business_dictionary.py` 重新提取。模拟数据使用字典中的原表名，不创建新的 `demo_*` 或简化销量表。

## 配置与初始化

先在项目根目录 `.env` 填写本机连接信息，不要将密码提交到 Git：

```dotenv
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=你的账号
MYSQL_PASSWORD='你的密码'
MYSQL_DATABASE=rag_qa_system
```

显式配置的 `DATABASE_URL` 优先于 `MYSQL_*`。如已配置旧 SQLite URL，需移除该项或改成目标 MySQL URL。建议使用本项目专用空库；初始化工具拒绝往已有业务记录的库中混入模拟数据。

```powershell
.\.venv\Scripts\python.exe -m pip install PyMySQL==1.1.2
.\.venv\Scripts\python.exe scripts/seed_mysql_business.py --create-database
.\.venv\Scripts\python.exe run_api.py
# 在另一个终端运行
.\.venv\Scripts\python.exe run_frontend.py
```

造数默认覆盖截至昨天的90天，包含6个店铺、6个主播、12个商品、直播商品明细、视频销售明细和可确定口径的加工数据。同一天包含多场直播；另有未结束直播，用于检验“最新一场”不会错误选成整天或正在进行的场次。可用 `--end-date YYYY-MM-DD --days 90` 固定演示日期。造数不会在每次启动时执行；再次执行只核对现有初始化记录和数量，不覆盖或刷新已有数据。

应用启动时在同一个 MySQL 库建立问答、反馈、文档、任务等系统表；这些不是字段字典中的经营数据表。旧 SQLite 文件保留，历史问答和文档登记不会自动迁移。已有 Chroma 向量仍在本地文件中；这次 MySQL 改动不迁移向量库，也不宣称原文档权限/版本已迁入新主库。

## 示例输入

浏览器打开 `http://localhost:8501`，进入问答页、新建会话：

> 查询各店铺的直播销售额排名。

样例店铺：清研美妆旗舰店、日光美妆店、简居生活旗舰店、山野食品旗舰店、云衣服饰旗舰店、简居家居店。

主播查询可用姓名或数值ID：陈洁KiKi为1001，美妆阿沐为1002，其余依次为1003至1006。

> 分析主播1001近30天直播和视频表现，并给出选品建议。

店铺直播复盘的专用入口和固定查数流程已移除。经营分析沿用现有通用工具流程：检索工具说明，模型在已有工具中选择调用，程序保留账号校验和补查逻辑。目前工具主要围绕主播，无法确认主播时会要求提供主播名称或ID；本次未新增通用店铺动态规划能力。MySQL 模拟数据不变，历史店铺报告仍可展示。

MySQL模式下，事实查询通过字段字典核对真实表结构，再生成单条MySQL SELECT；执行前检查表和字段范围、函数、结果上限，执行时使用只读事务及查询超时。默认只给原始业务表，明确指定加工表时才使用对应加工表，避免跨层重复求和。SQLite旧演示逻辑仅保留兼容测试，不作为本机MySQL的数据源。PostgreSQL企业模板的经营查询不属于本次适配范围。

## 字典缺失与模拟约定

字典共64张表：49张有字段清单，可建立物理表；15张只有表名，没有可用字段，保留为待补充，不创建虚假字段。49张中，`dws_livebigtable` 的全部字段类型未提供；`ads_live_product_sale_info.sales` 要求非空却没有含义定义。这两张保留空表，其他47张生成模拟记录。未确认类型的字段使用可空TEXT保留列名，未确认含义的字段不提供给自动SQL，造数时保持NULL。未确认表清单和字段在 `business_schema.json` 中可直接查阅。

`business_datasets` 是系统新增的模拟数据登记表，记录字典摘要、版本、数据量和以下约定，不伪装成原始业务表：

- `jl_user.shop_id` 在模拟数据中表示自播店铺，`live_base.id` 对应其他直播表的 `room_id`。真实库的关系仍需真实数据验证。
- `live_product_sale.number` 在模拟数据中作为成交件数；`live_sales_info.sales` 为人民币元；BIGINT整数销售额使用整数元固定价格生成。
- 商品明细金额以 `number × min_price` 估算，不能视为真实成交额；整场销售额以直播销售表为准。
- DATETIME使用北京时间，整数时间使用Unix秒，BIGINT日期标记使用YYYYMMDD；同名字段按各表类型分别处理。
- 数组在MySQL中存为JSON；MySQL不能直接容纳的超长VARCHAR改用TEXT，原始字典类型仍完整保留。
- `dws_live_product_info.sales` 按该表字典“销售均价”生成，不能当作整场销售额汇总。字段同名不代表统计口径相同。
- 没有订单、退款、成本和利润数据，不生成净收入、利润、购买转化率或外部行业结论。
- 经营模拟数据目前是共享样本，不等同于真实业务的租户级数据权限。

原备份工具仅支持SQLite/PostgreSQL，不支持MySQL；本机MySQL需使用MySQL专用备份工具。本次没有修改备份流程或删除旧数据。
