# 单轮模型对比测试

测试类是 `scripts/compare_llm_providers.py` 中的 `LLMComparisonTest`。每轮同时调用三家，每家一次，使用完全相同的系统提示词、上下文和问题，跳过 RAG 检索，比较模型 API 的速度和完整回答。

## 1. 直接在 JSON 配置

打开 `scripts/llm_compare.example.json`，在每个服务中填写 `model`、`base_url` 和 `api_key`。测试工具只读取 JSON 中这三项，不读取项目 `.env` 或系统环境变量，也无需单独的密钥文件。三家的密钥都需要在 JSON 中填写。

文件格式如下（空字符串处填写对应服务的 API 密钥，模型和地址按你的账号实际支持情况填写）：

```json
{
  "system_prompt": "你是一个专业的企业知识助手，请根据以下上下文回答问题。证据不足时回复“根据现有知识库无法回答该问题”。保持客观、简洁。\n【上下文信息：虚构演示制度，仅用于测试】\n连续工作满1年不满10年的，年假5天；满10年不满20年的，年假10天；满20年及以上的，年假15天。法定节假日和休息日不计入年假。",
  "question": "年假放多少天",
  "providers": [
    {
      "name": "元宝对应混元 API（非网页版）",
      "model": "hunyuan-turbos-latest",
      "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
      "api_key": ""
    },
    {
      "name": "GPT-6",
      "model": "gpt-6.1-sol",
      "base_url": "https://api.tangguo.xin/v1",
      "api_key": ""
    },
    {
      "name": "千问",
      "model": "qwen3.8-max",
      "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
      "api_key": ""
    }
  ]
}
```

这份文件包含明文密钥，已被 `.gitignore` 精确忽略，不会被常规 Git 提交收录。新环境如果没有该文件，可以按上面的完整示例创建。不要分享填有密钥的 JSON；如果使用 `--config` 指定另一个文件，也应将那个文件加入 Git 忽略。

元宝项目前是腾讯混元 API 示例，不等于元宝网页版。若在元宝选择了 DeepSeek，请把对应项改为所选 DeepSeek 服务的实际地址、模型和密钥。GPT 的 `gpt-6.1-sol` 是否可用，由所填 API 服务决定；千问也以你填写的模型及入口为准。

可选项保持原有用法：服务不支持流式用量参数时，可添加 `"include_usage": false`；需要额外模型参数时，可添加 `"extra_body": {"enable_thinking": false}`，仅填写该服务支持的参数。默认不强制温度或思考模式，三家默认行为可能不同。

## 2. 运行一次测试

在项目根目录的 PowerShell 中，先检查配置，这一步不调用模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_llm_providers.py --check
```

填好三家密钥后直接运行，每家只调用一次，失败不自动重试：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_llm_providers.py
```

缺少任何一家密钥时，本次不会调用任何服务。JSON 的 `system_prompt` 填系统提示词和上下文，`question` 填用户问题。也可以把完整系统提示词及上下文保存到 UTF-8 文件，并这样测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_llm_providers.py --prompt-file data/llm_compare_prompt.txt --question "年假放多少天" --timeout 60
```

真实运行会将相同输入发送给三家配置的服务并产生 API 费用；测试业务资料前请确认可以发给这些服务。结果只在控制台显示，不自动保存业务内容。

## 3. 怎样看结果

- **首个流片段**：从开始 API 调用到 SDK 收到第一个片段，片段可能只有角色、空内容或单独的思考信息。
- **首段正文**：从开始 API 调用到收到第一段非空白答案正文；单独的思考字段不算正文。包含网络和模型服务等待，不包含 RAG 检索或页面显示。
- **总耗时**：调用结束或失败的时间；截断、断线或只有思考没有正文的回答不会标为完成。
- **输入/输出 Token**：服务提供的用量，缺失时显示“未提供”。不同模型的 Token 划分可能不同。
- **完整回答**：逐家打印，人工核对关键事实、遗漏和无关内容，不把关键词命中率当成准确率。

单次结果受服务负载、网络和模型默认参数影响，适合快速对照，不能作为稳定性能排名。实际模型名称取自服务返回，不能独立验证服务底层使用了什么模型。控制台不打印密钥，也不回显供应商可能含敏感信息的原始错误正文。

无需真实密钥的离线验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_llm_comparison.py -q
```

接口参考：[腾讯混元 OpenAI 兼容接口](https://cloud.tencent.com/document/product/1729/111007)、[千问 OpenAI 兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)、[OpenAI 流式响应](https://developers.openai.com/api/docs/guides/streaming-responses)。
