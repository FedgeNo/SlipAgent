### Continuation After Tools

`retained_user_prompts` contains the last request. Use that request and the previous turn's tool calls and attached results in `history` to choose this turn's action. If those results fulfill the request, report the outcome with no tool calls. Otherwise, request tools needed to advance it.
