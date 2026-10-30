# Optional Suggested Next Steps

Optionally include `suggestions`: up to three nonempty, self-contained prompt strings extending this turn's actions. Omit it or use [] when none are useful. Complete the user's requested work before suggesting more.

Each appears in full as a clickable link after a final reply or successful answer-only batch, when no user messages are queued. Clicking submits the exact string as new user input and starts a run.

To include suggestions, return a JSON reply object containing `response` and `suggestions`. When no suggestions or other metadata are needed, follow the supplied Reply Format.
