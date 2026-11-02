#!/usr/bin/env python3
"""Requests the server must refuse with HTTP 400 instead of serving loosely.

Each malformed tool_choice / response_format is rejected before any
generation starts, with an OpenAI style error body
(``{"error": {"message", "type": "invalid_request_error", "code": 400}}``),
for streaming requests too. A good request afterwards must still work:
a rejection must not leave the server in a bad state.

Usage::

    python3 test_errors.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, case, expect
from fixtures import WEATHER_TOOLS, function_choice, json_schema_format, user


def expect_400(api: Api, fields: dict[str, Any], message_part: str) -> None:
    """Non-streaming and streaming both refuse, naming the problem."""
    body = api.body(user("hi"), max_tokens=8, **fields)
    resp = api.post(body)
    expect(resp.status == 400, f"expected HTTP 400, got {resp.status}: {resp.body}")
    err = resp.body.get("error")
    expect(isinstance(err, dict), f"error body is not an object: {resp.body}")
    expect(err.get("type") == "invalid_request_error", f"error type is {err.get('type')!r}")
    expect(message_part in str(err.get("message")), f"message {err.get('message')!r} lacks {message_part!r}")

    s = api.stream(body)
    expect(s.status == 400, f"streaming: expected HTTP 400, got {s.status}")
    expect(not s.chunks, "streaming: chunks were sent before the rejection")


@case
def bad_tool_choice_string(api: Api) -> None:
    """An unknown tool_choice string."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": "sometimes"}, "tool_choice must be")


@case
def bad_tool_choice_object(api: Api) -> None:
    """A tool_choice object without function.name."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": {"type": "function"}}, "tool_choice must be")


@case
def required_without_tools(api: Api) -> None:
    """required with no tools to call."""
    expect_400(api, {"tool_choice": "required"}, "has no tools")


@case
def function_without_tools(api: Api) -> None:
    """A forced function with no tools at all."""
    expect_400(api, {"tool_choice": function_choice("get_weather")}, "has no tools")


@case
def unknown_function(api: Api) -> None:
    """A forced function that is not among the tools."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": function_choice("launch_rocket")},
               "not in tools")


@case
def bad_response_format_type(api: Api) -> None:
    """An unknown response_format type."""
    expect_400(api, {"response_format": {"type": "xml"}}, "response_format type must be")


@case
def json_schema_without_schema(api: Api) -> None:
    """json_schema without its schema."""
    expect_400(api, {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
               "needs json_schema.schema")


@case
def uncompilable_schema(api: Api) -> None:
    """A schema the grammar compiler cannot build: a dangling $ref."""
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}, "required": ["a"]}
    expect_400(api, {"response_format": json_schema_format("bad", schema)}, "cannot build the decoding grammar")


@case
def null_options_are_unset(api: Api) -> None:
    """Options sent as null, as some clients do for unset ones, are not an error."""
    body = api.body(user("Say hi."), max_tokens=16, tools=WEATHER_TOOLS, tool_choice=None,
                    parallel_tool_calls=None, response_format=None, reasoning_effort=None,
                    presence_penalty=None, stream=None, max_completion_tokens=None)
    flm_api.check_message_contract(api.ok(body))


@case
def server_still_serves(api: Api) -> None:
    """After the rejections a normal request still succeeds."""
    flm_api.check_message_contract(api.ok(api.body(user("Say hi."), max_tokens=16)))


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
