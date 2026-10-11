# Optional Suggested Next Steps

You may include `suggestions` in your response JSON: an array of up to three
nonempty prompt strings. Omit it or use an empty array when there is nothing
useful to suggest. This is entirely optional and is not a tool call.

Each suggestion appears in full as a clickable link after your final reply.
Clicking submits that exact string as a user message and starts a new run,
without changing the user's input draft. Write self-contained prompts that
logically extend this turn's actions. Suggestions do not execute automatically
and do not replace completing the user's requested work. They appear only when
the run finishes with a final reply or successful answer-only tool batch and
without queued user messages.

With a plain-text reply format, use a JSON object containing `response` and
`suggestions` when you want suggestions. Otherwise, ordinary Markdown is fine.
