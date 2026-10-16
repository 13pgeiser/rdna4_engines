#!/bin/sh
# The four hardest tool-call shapes to parse (an array argument, two calls in one reply, prose
# before a call, plain chat with tools available) against a speculating container such as the
# DSpark one (scripts/mcserve.sh); speculation must not disturb the XML tool grammar.
# finish_reason tool_calls shows the server parsed the calls; a call returned as prose in
# `content` is one the parser did not close.
# Attaches to a running server on P= (default 8100).
P=${P:-8100}
T='[{"type":"function","function":{"name":"edit","description":"edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}}}}},"required":["path","edits"]}}},{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},{"type":"function","function":{"name":"write","description":"write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]'
ask() { # $1 label  $2 user message
  r=$(curl -s -m 180 http://127.0.0.1:$P/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":$2}],\"tools\":$T,\"max_tokens\":2500,\"temperature\":0}")
  echo "$r" | jq -c --arg l "$1" '{test:$l, finish:.choices[0].finish_reason,
      calls:[.choices[0].message.tool_calls[]?|{n:.function.name,a:(.function.arguments|tostring|.[0:90])}],
      leak:(.choices[0].message.content//""|test("<function|<param|CDATA")),
      content:((.choices[0].message.content//"")|.[0:70])}'
}
ask "edit-array-arg"  '"In /work/t.py change the text def minimise: to def minimax: -- use the edit tool."'
ask "two-calls"       '"Run ls /work and then run pwd, using the bash tool for each."'
ask "prose-then-call" '"Create /work/h.py containing print(\"hello\"). Explain what you will do first, then do it."'
ask "plain-chat"      '"In one sentence, what is a hash map?"'
