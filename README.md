# DS2Api

把 DeepSeek 网页版（chat.deepseek.com）包装成 **OpenAI 兼容 API** 的纯 Python 实现。

本项目是 [`NIyueeE/ds-free-api`](https://github.com/NIyueeE/ds-free-api)（Rust，GPL-3.0-only）的移植版本，
协议细节、默认值、错误码均以原项目源码为准。

## 与原项目的差异

| 项 | 原项目 | 本项目 |
|---|---|---|
| 对外协议 | OpenAI + Anthropic + Responses | **仅 OpenAI 兼容** |
| 工具调用 | `<|tool▁calls▁begin|>` JSON 协议 | **DSML 为主**，兼容旧协议 |
| 语言/框架 | Rust / axum | Python / FastAPI |
| TLS 指纹 | wreq（BoringSSL） | curl_cffi（`impersonate="chrome136"`） |
| PoW | wasmtime | wasmtime-py（同一份 wasm） |
| 管理面板 | Vite 前端 + admin API | admin API（前端待补） |

### 有意偏离原实现的地方

1. **`session_policy` 默认 `reuse`** —— 与原项目 `src/config.rs` 一致。原
   `config.example.toml` 的注释写的是 `per_request`，那是过期文案。
2. **`default_search_enabled` 默认 `False`** —— 同样是代码为准，注释过期。
3. **明文 HTTP 强制 HTTP/1.1** —— 仅测试环境触发；生产走 https 时保留浏览器
   的 HTTP/2 指纹。
4. **删除了两个导航形态请求头** —— curl_cffi 的 `chrome136` 会加
   `Sec-Fetch-User` 和 `Upgrade-Insecure-Requests`，真实前端是 XHR 调用，不带
   这两个，必须显式传 `None` 移除。

## 安装

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Windows
# .venv/bin/pip install -e ".[dev]"       # Linux / macOS
```

要求 Python ≥ 3.11。Windows 上 `curl_cffi` 需要 Selector 事件循环，程序已自动设置。

## 配置

复制 `config.example.toml` 为 `config.toml`，至少填三项：

```toml
[server]
host = "127.0.0.1"
port = 22217

[[ds_core.accounts]]
email = "you@example.com"
password = "your-password"
device_id = "从浏览器复制的数美设备指纹"

[[api_keys]]
key = "sk-your-key"
description = "本地开发"
```

`device_id` 获取方式：登录 chat.deepseek.com，在 DevTools → Network 里查看
`POST /api/v0/users/login` 的请求体，或控制台执行 `SMSdk.getDeviceId()`。
**不能伪造**，伪造会返回 `RISK_DEVICE_DETECTED`。每个账号建议独立设备指纹。

## 运行

```bash
python -m ds2api -c config.toml
```

环境变量：`DS_CONFIG_PATH`（配置路径）、`DS_DATA_DIR`（stats.json 与 wasm 缓存）、
`DS_LOG`（日志级别）、`DS_WASM_PATH`（离线 wasm 文件）。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | 聊天补全（流式 / 非流式 / 工具 / 内联文件） |
| GET | `/v1/models` | 模型列表 |
| GET | `/v1/models/{id}` | 模型详情 |
| GET | `/health` | 健康检查 |
| * | `/admin/api/*` | 管理接口（JWT 保护，`setup`/`login` 除外） |

```bash
curl http://127.0.0.1:22217/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-default","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 工具调用（DSML）

上游没有原生 function calling。请求里带 `tools` 时，工具定义会以 DSML 规范注入
system prompt，模型用 DSML 标记回话，网关解析后还原成标准 `tool_calls`：

```text
<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>
```

- `string="true"` 值是原始文本，不转义；`string="false"` 值是 JSON。
- 工具名可用 `namespace::name` 限定，返回时还原为裸名。
- 兼容 V4 无空格写法、`<||DSML||>` ASCII 变体、旧版
  `<|tool▁calls▁begin|>` 协议与裸 JSON 数组。
- 工具结果按 DSML 语义以 `<tool_result>` 包进用户回合（无 `tool` 角色）。

## 不支持

- Anthropic `/anthropic/v1/*`、Responses `/v1/responses`、`Idempotency-Key`
- `n > 1`、多候选、`audio` / `modalities`
- `temperature` / `top_p` / `max_tokens` / `stop` 透传（上游无对应字段；
  `stop` 在本地截断）
- 多进程（账号池与会话缓存是进程内状态，固定 `workers=1`）

## 测试

```bash
pytest -q                 # 全部（单元 + 端到端，假上游）
pytest tests/unit -q      # 纯函数
pytest tests/e2e -q       # 真 uvicorn + 假上游
```

Phase 0 的三个验证脚本在 `tools/`，结论见 `docs/spike-report.md`。

## 许可

GPL-3.0-only，与上游项目一致。本项目为衍生作品，保留原始项目的协议实现与
默认值以维持行为兼容。
