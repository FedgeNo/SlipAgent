"""Shared colors for ANSI output, input styling, and transcript recognition."""

USER_COLOR = "#66ff66"
ERROR_COLOR = "#ff6666"
MUTED_COLOR = "#aaaaaa"


def foreground_code(color: str) -> str:
    """Convert a six-digit palette color to an ANSI true-color foreground."""
    channels = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    return "38;2;" + ";".join(str(channel) for channel in channels)
