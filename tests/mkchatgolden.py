"""mkchatgolden.py -- reference renders of every shipped chat template, for tests/chat_test.cpp.

Reads tests/data/chat/golden/cases.json and renders each case through each model's
tests/data/chat/<model>/chat_template.jinja with Python's jinja2 (3.1) set up as transformers'
apply_chat_template sets it up (transformers/utils/chat_template_utils.py: an
ImmutableSandboxedEnvironment with trim_blocks and lstrip_blocks, the loopcontrols extension, a
`tojson` filter over json.dumps with ensure_ascii=False, raise_exception and strftime_now). When
transformers is importable its own render_jinja_template is used instead, and the two agree.

The cases hold requests as the server receives them; this script turns each into what a
transformers caller passes: tool-call arguments as objects, and a media part as the template's own
`{"type": "image"}` part. Every render passes enable_thinking, bos_token and eos_token, which the
engine always sets.

Writes tests/data/chat/<model>/golden/<case>.txt, the rendered prompt, or <case>.err, the message
of a template that refused the request. Run from anywhere: python3 tests/mkchatgolden.py
"""
import copy
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(HERE, "data", "chat")


def hf_environment():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def renderer():
    try:
        from transformers.utils.chat_template_utils import render_jinja_template
    except ImportError:
        render_jinja_template = None
    if render_jinja_template is not None:
        def render(source, messages, tools, add_generation_prompt, **kw):
            out, _ = render_jinja_template([messages], tools=tools, chat_template=source,
                                           add_generation_prompt=add_generation_prompt, **kw)
            return out[0]
        return render, "transformers render_jinja_template"
    env = hf_environment()

    def render(source, messages, tools, add_generation_prompt, **kw):
        return env.from_string(source).render(messages=messages, tools=tools, documents=None,
                                              add_generation_prompt=add_generation_prompt, **kw)
    import jinja2
    return render, "jinja2 %s" % jinja2.__version__


def hf_messages(messages):
    """The request as a transformers caller writes it."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
        if isinstance(m.get("content"), list):
            m["content"] = [{"type": "image"} if p.get("type") == "media_marker" else p
                            for p in m["content"]]
    return out


def main():
    import jinja2
    with open(os.path.join(CHAT, "golden", "cases.json"), encoding="utf-8") as f:
        spec = json.load(f)
    render, how = renderer()
    for model, info in spec["models"].items():
        with open(os.path.join(CHAT, model, "chat_template.jinja"), encoding="utf-8") as f:
            source = f.read()
        out_dir = os.path.join(CHAT, model, "golden")
        os.makedirs(out_dir, exist_ok=True)
        for name in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, name))
        for case in spec["cases"]:
            if "models" in case and model not in case["models"]:
                continue
            kw = dict(case.get("kwargs", {}))
            kw["enable_thinking"] = case.get("enable_thinking", True)
            kw["bos_token"] = info["bos"]
            kw["eos_token"] = info["eos"]
            tools = spec["tools"] if case.get("tools") else None
            try:
                text, ext = render(source, hf_messages(case["messages"]), tools,
                                   case.get("add_generation_prompt", True), **kw), ".txt"
            except jinja2.exceptions.TemplateError as e:
                text, ext = str(e), ".err"
            with open(os.path.join(out_dir, case["name"] + ext), "w", encoding="utf-8",
                      newline="") as f:
                f.write(text)
    print("rendered with %s" % how)


if __name__ == "__main__":
    sys.exit(main())
