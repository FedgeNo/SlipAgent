"""Shared colors for ANSI output, input styling, and transcript recognition."""

USER_COLOR = "#00ff00"
USER_TEXT_COLOR = "#ffffff"
USER_BACKGROUND_COLOR = "#004000"
USER_PROMPT_STYLE = USER_TEXT_COLOR + " bg:" + USER_BACKGROUND_COLOR
ERROR_COLOR = "#ff6666"
MUTED_COLOR = "#aaaaaa"
THOUGHT_COLOR = "#999999"


def foreground_code(color: str) -> str:
    """Convert a six-digit palette color to an ANSI true-color foreground."""
    channels = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    return "38;2;" + ";".join(str(channel) for channel in channels)
