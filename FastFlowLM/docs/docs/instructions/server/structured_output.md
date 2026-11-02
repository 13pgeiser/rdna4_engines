---
layout: docs
title: Structured Output & tool_choice
nav_order: 10
parent: Local Server (Server Mode)
---

# 🧩 Structured Output & `tool_choice` (Constrained Decoding)

Small local models are good at following a format *most* of the time. Constrained decoding makes it *all* of the time. Before every token is sampled, FLM works out which tokens can still lead to valid output and rules out all the others. A model can't write a misspelled tool name, an argument of the wrong type, a missing required field, or a stray Markdown fence around your JSON, because those tokens are never allowed.

You control it with the standard OpenAI request fields:

| Field | Values | What FLM guarantees |
|---|---|---|
| `tool_choice` | `"auto"` (default), `"none"`, `"required"`, `{"type": "function", "function": {"name": …}}` | Whether a call happens, which function, and that its arguments match the tool's `parameters` schema |
| `parallel_tool_calls` | `true` (default), `false` | At most one call per turn when `false` |
| `response_format` | `{"type": "text"}`, `{"type": "json_object"}`, `{"type": "json_schema", "json_schema": {"schema": …}}` | `content` is valid JSON, matching the schema if one is given |

It works the same with and without `stream`, and with or without thinking. The grammar engine is [xgrammar](https://github.com/mlc-ai/xgrammar).

> 📘 How reasoning, content and tool calls come back in the response is covered in [Reasoning, Tool Calls & Streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/).

---

# 📑 Table of Contents

- [What the Constraint Guarantees, and What It Doesn't](#-what-the-constraint-guarantees-and-what-it-doesnt)
- [tool_choice](#-tool_choice)
- [parallel_tool_calls](#-parallel_tool_calls)
- [Tool Schemas and `strict`](#-tool-schemas-and-strict)
- [response_format](#-response_format)
- [Use Cases](#-use-cases)
  - [Extract Data into a Pydantic Model](#-extract-data-into-a-pydantic-model)
  - [Classify with an Enum](#️-classify-with-an-enum)
  - [Quick JSON Mode](#-quick-json-mode)
  - [Force One Specific Function](#-force-one-specific-function)
  - [Route a Request to One of Several Tools](#-route-a-request-to-one-of-several-tools)
  - [Answer Without Tools on the Last Turn](#-answer-without-tools-on-the-last-turn)
  - [Think First, Then Answer in JSON](#-think-first-then-answer-in-json)
  - [LangChain](#-langchain)
  - [curl](#-curl)
- [Errors](#-errors)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## ✅ What the Constraint Guarantees, and What It Doesn't

**Guaranteed** (when the request ends with `finish_reason` `"stop"` or `"tool_calls"`):

- Tool names are one of the tools you declared.
- Tool arguments are well-formed and follow the tool's `parameters` schema: types, `required` fields, `enum` values, and array `items`.
- `response_format` content parses as JSON and follows the schema.
- When thinking is off, no reasoning block is generated.

**Not guaranteed:**

- **Values that make sense.** The model can still put `"Paris"` where you wanted `"Tokyo"`. The constraint shapes the output; it doesn't check facts.
- **Complete output on `"length"`.** If `max_tokens` runs out mid-JSON, the JSON is cut off. Give structured requests enough room, and more when thinking is on.
- **Silence around calls under `"auto"` / `"required"`.** Those modes allow text before, between and after calls (see below). Only the forced-function form allows nothing but the call.

> 💡 Describe the format in your prompt as well. The constraint stops invalid output, but the model picks better values when it knows what you want.

---

## 🎛️ `tool_choice`

| Value | What the model may write |
|---|---|
| `"auto"` *(default)* | Free text, or tool calls, or both. Once the model **starts** a call, the rest of that call is constrained: the name must be a declared tool and the arguments must match its schema. |
| `"none"` | Text only. The tools are **left out of the prompt**, so the model doesn't know they exist, and the call opener token is banned. |
| `"required"` | **At least one** tool call. Text is still allowed around the calls. Pair with `parallel_tool_calls=False` for exactly one. |
| `{"type": "function", "function": {"name": "f"}}` | **Exactly one** call to `f` and nothing else in `content` (the model may still reason first when thinking is on). |

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")

resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    tool_choice="required",          # or "auto", "none",
                                     # or {"type": "function", "function": {"name": "get_weather"}}
    extra_body={"think": False},
)
```

Notes:

- `"required"` and the function form need a model whose format has a tool call syntax (every format except `content_only`, see [the format table](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/#️-supported-model-formats)). On other models they're rejected with HTTP 400 rather than silently ignored.
- They're also rejected on models too small to be shown tools (`qwen3:0.6b`, `qwen3:1.7b`, `qwen3.5:0.8b`). These leave `tools` out of the prompt, so a forced call would have arguments made up for a tool the model never saw.
- `"auto"` on a model without a tool call syntax, or one that isn't shown tools, is accepted, but nothing is constrained.
- With `"required"`, small models sometimes keep writing after their call, for example a made-up answer to their own question. Look at `tool_calls`. `content` may hold this extra text.

---

## 🧵 `parallel_tool_calls`

| Value | Effect |
|---|---|
| `true` *(default)* | The model may make several calls in one turn. |
| `false` | Generation stops after the first complete call. |

`parallel_tool_calls=False` with `tool_choice="required"` means **exactly one call**: a clean way to make a model pick one action.

---

## 📐 Tool Schemas and `strict`

The tool's `function.parameters` is the schema the arguments must follow.

- **Schemas are enforced by default.** In FLM every tool behaves as if `"strict": true`. Set `"strict": false` on a tool to only require well-formed JSON arguments.
- A tool with **no `parameters`** gets any well-formed JSON arguments.
- An array without `items` accepts any items.
- **Qwen3.5+ booleans** (`qwen_xml` format): the model's template writes Python-style `True`/`False`. Both spellings are accepted and come back as JSON `true`/`false`.
- **Gemma4 and LFM2** use FLM's own argument grammars, written for each model's native call syntax, instead of a full JSON-schema engine. Top-level arguments get their declared types, `required` fields, string `enum`s and array `items` enforced. Arguments are written in one fixed order (alphabetical for Gemma4, declaration order for LFM2). **Nested objects** are only held to valid syntax, not to their sub-schema. Numbers are limited to 16 digits so a model can't pad one forever.

---

## 📄 `response_format`

### JSON mode

```python
resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Give me a JSON object with keys color and hex for red."}],
    response_format={"type": "json_object"},
)
print(resp.choices[0].message.content)   # {"color": "red", "hex": "#FF0000"}
```

`content` is a JSON **object**: any keys, any values.

### JSON schema

```python
resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Extract: Alice is 31 and lives in Berlin."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",                     # accepted, not used
            "strict": True,                       # accepted; the schema is always enforced
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
                "additionalProperties": False,
            },
        },
    },
    extra_body={"think": False},
)
print(resp.choices[0].message.content)   # {"name": "Alice", "age": 31, "city": "Berlin"}
```

Details:

- The schema supports what xgrammar supports. That covers types, `properties`, `required`, `enum`, `const`, `items`, `anyOf`, `$defs`/`$ref` (so Pydantic's nested models work), string `pattern`s and number bounds. A schema xgrammar can't compile is rejected with a 400 that names the problem.
- **Whitespace is bounded.** Up to 8 blanks before the JSON and a limited amount of indentation inside it. This stops a constrained model from padding with blank lines forever.
- **Thinking still works.** With thinking on, the model reasons first, freely, into `reasoning_content`. Only `content` is constrained.
- **`response_format` and tools.** With `tool_choice="auto"`, `response_format` wins: the answer is JSON and the model can't call tools. With `"required"` or a forced function, the call wins and `response_format` is ignored. Use separate requests if you need both.
- Generation ends when the JSON is complete: `finish_reason` is `"stop"`.

---

## 💡 Use Cases

All examples assume:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")
MODEL = "qwen3:4b"
```

### 🧾 Extract Data into a Pydantic Model

Let Pydantic write the schema and validate the result:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    qty: int

class Order(BaseModel):
    customer: str
    items: list[Item]

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Bob wants 2 apples and 3 pears."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "Order", "schema": Order.model_json_schema()},
    },
    extra_body={"think": False},
)
order = Order.model_validate_json(resp.choices[0].message.content)
print(order)
# customer='Bob' items=[Item(name='apples', qty=2), Item(name='pears', qty=3)]
```

> Newer OpenAI SDKs also have `client.chat.completions.parse(response_format=Order, …)`. It sends the same `json_schema` request, so it works with FLM too.

### 🏷️ Classify with an Enum

The model can only write one of the labels:

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Classify the sentiment: I loved this movie!"}],
    response_format={"type": "json_schema", "json_schema": {"name": "sentiment", "schema": {
        "type": "object",
        "properties": {"sentiment": {"type": "string",
                                     "enum": ["positive", "neutral", "negative"]}},
        "required": ["sentiment"],
        "additionalProperties": False,
    }}},
    max_tokens=30,
    extra_body={"think": False},
)
label = json.loads(resp.choices[0].message.content)["sentiment"]   # "positive"
```

### 🧱 Quick JSON Mode

When you only need *some* JSON object:

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": "Reply with a JSON object with keys 'title' and 'tags'."},
              {"role": "user", "content": "Summarize: NPUs make local LLMs fast and efficient."}],
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)
```

### 🎯 Force One Specific Function

Name the function, and the response is exactly one call to it with valid arguments and no text. This is a common way to do *extraction through a tool definition*:

```python
tools = [{"type": "function", "function": {
    "name": "create_event",
    "description": "Add an event to the calendar",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "attendees": {"type": "array", "items": {"type": "string"}},
    }, "required": ["title", "date"]},
}}]

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Lunch with Ana and Raj on 2026-11-03."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "create_event"}},
    extra_body={"think": False},
)
call = resp.choices[0].message.tool_calls[0]
print(call.function.name, json.loads(call.function.arguments))
# e.g. create_event {'title': 'Lunch with Ana and Raj', 'date': '2026-11-03', 'attendees': ['Ana', 'Raj']}
```

### 🔀 Route a Request to One of Several Tools

`"required"` plus `parallel_tool_calls=False` makes the model pick **exactly one** of your tools. Use it as an intent router:

```python
router_tools = [
    {"type": "function", "function": {"name": "search_docs",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_ticket",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]}},
            "required": ["summary", "priority"]}}},
    {"type": "function", "function": {"name": "small_talk",
        "parameters": {"type": "object", "properties": {}}}},
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "The server is down and customers can't log in!"}],
    tools=router_tools,
    tool_choice="required",
    parallel_tool_calls=False,
    extra_body={"think": False},
)
call = resp.choices[0].message.tool_calls[0]
print(call.function.name, call.function.arguments)
# e.g. open_ticket {"summary": "Server down, customers cannot log in", "priority": "high"}
```

### 🔚 Answer Without Tools on the Last Turn

In an agent loop, once you have enough tool results, `tool_choice="none"` makes the model write its final answer instead of calling yet another tool:

```python
messages.append({"role": "user", "content": "Thanks, now summarize what you found."})
final = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, tool_choice="none",
)
print(final.choices[0].message.content)
```

You can keep sending the same `tools` list. FLM leaves it out of the prompt for this request.

### 🤔 Think First, Then Answer in JSON

On reasoning models, thinking before a structured answer often improves the values. The thinking is free text and only the answer is constrained:

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "A train leaves at 14:35 and arrives at 17:10. "
                                          "How long is the trip? Answer as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "duration", "schema": {
        "type": "object",
        "properties": {"hours": {"type": "integer"}, "minutes": {"type": "integer"}},
        "required": ["hours", "minutes"], "additionalProperties": False}}},
    max_tokens=2048,                   # leave room for the thinking
    extra_body={"think": True},
)
msg = resp.choices[0].message
print("thought:", getattr(msg, "reasoning_content", "")[:200], "…")
print("answer:", json.loads(msg.content))      # {'hours': 2, 'minutes': 35}
```

### 🦜 LangChain

`ChatOpenAI` sends the same fields, so LangChain's structured output helpers work against FLM:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

llm = ChatOpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm", model="qwen3:4b",
                 extra_body={"think": False})

# via response_format json_schema
person = llm.with_structured_output(Person, method="json_schema").invoke("Alice is 31.")

# via a forced tool call
person = llm.with_structured_output(Person, method="function_calling").invoke("Alice is 31.")
```

### 🐚 curl

```bash
curl http://127.0.0.1:52625/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "think": false,
    "messages": [{"role": "user", "content": "Extract: Alice is 31 and lives in Berlin."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
      "required": ["name", "age", "city"], "additionalProperties": false}}}
  }'
```

---

## ❌ Errors

A request FLM can't serve as asked gets **HTTP 400** with an OpenAI-style error body, instead of being quietly downgraded:

```json
{"error": {"message": "tool_choice names function \"nope\" which is not in tools",
           "type": "invalid_request_error", "code": 400}}
```

In the OpenAI Python SDK this raises `openai.BadRequestError`.

| Request | Message |
|---|---|
| `tool_choice: "sometimes"` | `tool_choice must be "auto", "none", "required" or a function, got "sometimes"` |
| `tool_choice: "required"` with no `tools` | `tool_choice requires a tool call but the request has no tools` |
| Function form naming a tool that isn't in `tools` | `tool_choice names function "nope" which is not in tools` |
| `response_format: {"type": "xml"}` | `response_format type must be text, json_object or json_schema` |
| `json_schema` without `schema` | `response_format json_schema needs json_schema.schema` |
| A schema xgrammar can't compile (e.g. a dangling `$ref`) | `cannot build the decoding grammar: …` with xgrammar's reason |
| `"required"` / function form on a model without tool syntax | `this model's chat format (content_only) has no tool call syntax to constrain` |
| `"required"` / function form on a model too small to be shown tools | `this model does not support tool calling; tool_choice must be "auto" or "none"` |

Options sent as `null`, as some clients do for the ones they leave unset, are treated as unset, not as errors.

---

## ⚡ Performance

- **First constrained request:** FLM builds the grammar engine for the loaded model. It indexes the whole vocabulary once, and the log shows `Grammar engine ready in … ms` (about 0.3 s for Qwen3). Later requests reuse it.
- **Per request:** the request's grammar is compiled. That's quick for typical tool lists and schemas and grows with schema size.
- **Per token:** the allowed-token mask is computed on the CPU before each sample, which adds a small, fixed cost per token.
- **Unconstrained requests pay nothing.** A request with no `tools` and no `response_format`, or with `tool_choice="auto"` on a model without a tool call syntax, never touches the grammar engine.

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `content` is cut-off JSON and `finish_reason` is `"length"` | Raise `max_tokens`. With thinking on, budget for the reasoning too. |
| Values are valid but wrong | Improve the prompt and field `description`s, use a bigger model, or let it think first. |
| Server log says `Constrained decoding disabled for this request: …` | The grammar lost track of the output (rare, e.g. a template quirk) and FLM let the model finish unconstrained rather than produce garbage. Validate on your side too, and please report it with the log line. |
| 400 `has no tool call syntax to constrain` | The model has no tool format. Use a model labelled `tool-calling`, or use `response_format` instead. |
| `"required"` returned a call **and** some text | Expected: `"required"` only guarantees *at least one* call. Use the function form, or ignore `content`. |
