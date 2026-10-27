"""Mouse report mode requested from the terminal."""

import io

from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output

from slipagent.terminalmouse import limit_mouse_reports


def test_mouse_reporting_asks_for_clicks_and_drags_not_every_movement():
    sink = io.StringIO()
    output = Vt100_Output(sink, lambda: Size(rows=24, columns=100), term="xterm-256color", enable_cpr=False)
    limit_mouse_reports(output)
    output.enable_mouse_support()
    output.disable_mouse_support()
    output.flush()
    written = sink.getvalue()
    assert "\x1b[?1002h" in written and "\x1b[?1006h" in written
    assert "\x1b[?1003h" not in written
    assert "\x1b[?1002l" in written
