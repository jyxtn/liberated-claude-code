# Translation Layer

The proxy sits between Claude Code and model providers. It receives Anthropic-format requests on `/v1/messages` and must either pass them through or translate them depending on what the downstream provider accepts.

## Provider Categories

### Anthropic Pass-Through Providers

These providers accept Anthropic's native `/v1/messages` format directly. No translation needed.

| Provider | File | Endpoint |
|---|---|---|
| LM Studio | `providers/lmstudio/client.py` | `/v1/messages` |
| llama.cpp | `providers/llamacpp/client.py` | `/v1/messages` |
| Ollama Cloud | `providers/ollama/cloud.py` | `/v1/messages` |
| Ollama Local | `providers/ollama/local.py` | `/v1/messages` |

**Flow:** Request arrives as Anthropic format → forwarded as-is → response SSE events yielded directly.

**How to add a new pass-through provider:** Subclass `BaseProvider`, implement `stream_response()` to POST the raw Anthropic request body to the provider's `/v1/messages` endpoint, and yield SSE lines back.

### OpenAI-Compatible Providers

These providers accept OpenAI's `/v1/chat/completions` format. The proxy translates both directions.

| Provider | File | Translation |
|---|---|---|
| NVIDIA NIM | `providers/nvidia_nim/client.py` | Anthropic→OpenAI + OpenAI→Anthropic |
| OpenRouter | `providers/open_router/client.py` | Anthropic→OpenAI + OpenAI→Anthropic |
| Modal | `providers/modal/client.py` | Anthropic→OpenAI + OpenAI→Anthropic |
| OpenAI-Compat | `providers/openai_compat_generic/client.py` | Anthropic→OpenAI + OpenAI→Anthropic |

**Flow:**

1. Request arrives as Anthropic format
2. `_build_request_body()` converts to OpenAI format
3. `client.chat.completions.create()` opens streaming connection
4. OpenAI SSE chunks translated to Anthropic SSE events via `SSEBuilder`
5. Anthropic SSE events yielded back to Claude Code

**How to add a new OpenAI-compatible provider:** Subclass `OpenAICompatibleProvider`, implement `_build_request_body()`. The base class handles streaming, rate limiting, and SSE translation automatically.

## Key Files

### Request Translation (Anthropic → OpenAI)

- **`providers/common/message_converter.py`** — `AnthropicToOpenAIConverter` class and `build_base_request_body()` function
  - Converts Anthropic messages (content blocks, tool results, thinking) to OpenAI format
  - Converts Anthropic tool definitions to OpenAI `function` format
  - Converts Anthropic `system` field to OpenAI `system` message

### Response Translation (OpenAI → Anthropic)

- **`providers/common/sse_builder.py`** — `SSEBuilder` class
  - Builds Anthropic SSE events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`
  - Handles text, thinking, and tool call content blocks
  - Includes error event emission

### Provider Base Classes

- **`providers/base.py`** — `BaseProvider` with abstract `stream_response()` and `cleanup()`
- **`providers/openai_compat.py`** — `OpenAICompatibleProvider` with streaming loop, rate limiting, tool parsing, and SSE translation
  - Abstract method: `_build_request_body(request) -> dict`
  - Optional override: `_handle_extra_reasoning(delta, sse) -> Iterator[str]`

## Decision Flowchart

```
Does your endpoint accept /v1/messages (Anthropic format)?
├── Yes → Subclass BaseProvider
│          Implement stream_response() to POST and yield SSE
│          Examples: LMStudioProvider, LlamaCppProvider, OllamaCloudProvider
└── No  → Subclass OpenAICompatibleProvider
           Implement _build_request_body()
           Get translation for free
           Examples: NvidiaNimProvider, OpenRouterProvider, ModalProvider, OpenAICompatProvider
```

## Translation Details

### Message Conversion

| Anthropic | OpenAI |
|---|---|
| `role: "user"` with text content | `role: "user", content: "text"` |
| `role: "assistant"` with text content | `role: "assistant", content: "text"` |
| `content: [{type: "text", text: "..."}]` | `content: "..."` (single text) or array (multiple blocks) |
| `content: [{type: "thinking", thinking: "..."}]` | `content: "<thinking>...</thinking>"` (inline tags) |
| `content: [{type: "tool_use", ...}]` | `tool_calls: [{function: {name, arguments}}]` |
| `role: "user", content: [{type: "tool_result", ...}]` | `role: "tool", content: "...", tool_call_id: "..."` |
| `system: "You are..."` | `messages: [{role: "system", content: "..."}]` |

### Tool Definition Conversion

| Anthropic | OpenAI |
|---|---|
| `name` | `function.name` |
| `description` | `function.description` |
| `input_schema` | `function.parameters` |
| (top-level fields) | Wrapped in `{type: "function", function: {...}}` |

### Stop Reason Mapping

| OpenAI | Anthropic |
|---|---|
| `stop` | `end_turn` |
| `tool_calls` | `tool_use` |
| `max_tokens` | `max_tokens` |
| (length) | `max_tokens` |