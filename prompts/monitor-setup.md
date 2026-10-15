# Command Monitoring Setup

To watch a background job's output or completion, load `monitor_command` if needed with `load_tools({"names":["monitor_command"]})`. Start the job with `run_command` and `background=true`, then initialize a watch using the returned `job_id`: `{"action":"start","job_id":"ID_FROM_RESULT","pattern":"ready","timeout":120}`. Patterns are case-sensitive literal substrings. Omit `pattern` for completion; `repeat=true` watches repeated matches until the job ends or the watch expires. The default completes on the first match.

Start returns a `monitor_id` and records the initialization turn. It replays up to 65,536 retained UTF-8 bytes per stream, reports earlier omissions, then watches new output. Subsequent normal requests include active state and ready results. Continue independent authorized work or use `action="wait"` with `monitor_id` for up to 30 seconds.

Watch lifetime defaults to 30 seconds, capped at 600; the command timeout remains independent. Watches survive compatible reloads and end at process restart or session replacement. Use supplied live state to establish active watches.
