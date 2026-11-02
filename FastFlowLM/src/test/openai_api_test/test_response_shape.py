#!/usr/bin/env python3
"""Response shape of /v1/chat/completions, streamed and not.

Checks the contract every client relies on, whatever the model:

* a non-streaming message has the OpenAI fields, no raw template markup,
  and a finish_reason that agrees with its tool calls;
* every stream chunk carries exactly one kind of delta, one stream id,
  one finish chunk, then ``data: [DONE]``;
* usage is only streamed when ``stream_options.include_usage`` asks, as
  one last chunk with no choices;
* streaming and non-streaming produce the same message: the server
  parses both with one parser, so with greedy sampling they must agree
  (skipped when greedy decoding itself does not reproduce).

Usage::

    python3 test_response_shape.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import flm_api
from flm_api import Api, case, expect
from fixtures import WEATHER_TOOLS, function_choice, user

PROMPT = "In one short sentence, what is the capital of France?"


@case
def nonstream_message(api: Api) -> None:
    """A plain answer has role, string content and a usage block that adds up."""
    resp = api.post(api.body(user(PROMPT), max_tokens=64))
    expect(resp.status == 200, f"HTTP {resp.status}: {resp.error_message()}")
    expect(resp.body.get("object") == "chat.completion", f"object is {resp.body.get('object')!r}")
    choice = resp.body["choices"][0]
    msg = flm_api.check_message_contract(choice)
    expect(msg["content"].strip() != "", "content is empty")
    expect("tool_calls" not in msg, "no tools were offered, yet the message has tool_calls")
    usage = resp.body.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        expect(isinstance(usage.get(key), int), f"usage.{key} missing: {usage}")
    expect(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
           f"usage does not add up: {usage}")


@case
def stream_without_usage(api: Api) -> None:
    """Without include_usage no chunk carries a usage key."""
    s = api.ok_stream(api.body(user(PROMPT), max_tokens=64))
    folded = flm_api.check_stream_contract(s, include_usage=False)
    expect(folded.content.strip() != "", "streamed content is empty")
    expect(folded.finish_reason in ("stop", "length"), f"finish_reason {folded.finish_reason!r}")


@case
def stream_with_usage(api: Api) -> None:
    """include_usage puts usage: null on every chunk and the totals on a last, choice-less chunk."""
    s = api.ok_stream(api.body(user(PROMPT), max_tokens=64, stream_options={"include_usage": True}))
    folded = flm_api.check_stream_contract(s, include_usage=True)
    last = s.chunks[-1]
    expect(last["choices"] == [], "the usage chunk must be the last one, with no choices")
    for chunk in s.chunks[:-1]:
        expect(chunk["usage"] is None, "usage must be null on every chunk but the last")
    expect(folded.usage["completion_tokens"] > 0, f"usage reports no completion tokens: {folded.usage}")


@case
def length_finish(api: Api) -> None:
    """Running out of max_tokens reports finish_reason "length" in both modes."""
    body = api.body(user("Count from 1 to 100, separated by commas."), max_tokens=8)
    choice = api.ok(body)
    expect(choice["finish_reason"] == "length", f"non-stream finish_reason is {choice['finish_reason']!r}")
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    expect(folded.finish_reason == "length", f"stream finish_reason is {folded.finish_reason!r}")


@case
def stream_matches_nonstream(api: Api) -> None:
    """Greedy streaming and non-streaming answers are the same text."""
    body = api.body(user("Name three primary colors, comma separated."), max_tokens=48)
    flm_api.assert_parity(api, body, lambda m: (m["reasoning_content"], m["content"]))


@case
def stream_matches_nonstream_tools(api: Api) -> None:
    """With a forced call, both modes return the same call name and arguments."""
    body = api.body(user("What's the weather in Paris?"), tools=WEATHER_TOOLS,
                    tool_choice=function_choice("get_weather"))
    flm_api.assert_parity(
        api, body, lambda m: [(c["function"]["name"], c["function"]["arguments"]) for c in m["tool_calls"]])


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
