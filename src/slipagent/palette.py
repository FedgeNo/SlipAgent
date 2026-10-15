"""Shared colors for ANSI output, input styling, and transcript recognition."""

TEXT_COLOR = "#ffffff"
LINK_COLOR = "#83d6ef"
TOOL_COLOR = "#c4a7ef"
SUCCESS_COLOR = "#82d9ae"
WARNING_COLOR = "#f0c674"
USER_COLOR = SUCCESS_COLOR
USER_TEXT_COLOR = "#ffffff"
USER_BACKGROUND_COLOR = "#153c35"
CONTENT_BACKGROUND_COLOR = "#000000"
FOOTER_BACKGROUND_COLOR = "#17232d"
MENU_BACKGROUND_COLOR = "#272334"
FREE_MODEL_BACKGROUND_COLOR = "#17452f"
PANEL_HEADER_BACKGROUND_COLOR = "#373046"
EDITOR_BACKGROUND_COLOR = "#111c20"
CODE_BACKGROUND_COLOR = "#181c23"
INLINE_BACKGROUND_COLOR = "#28323b"
DIFF_ADDED_BACKGROUND_COLOR = "#122c22"
DIFF_REMOVED_BACKGROUND_COLOR = "#341e24"
USER_PROMPT_STYLE = USER_TEXT_COLOR + " bg:" + USER_BACKGROUND_COLOR
ERROR_COLOR = "#ff929b"
MUTED_COLOR = "#a5adb8"
THOUGHT_COLOR = "#a0a0a0"
THOUGHT_CODE_BACKGROUND_COLOR = "#242424"


def foreground_code(color: str) -> str:
    """Convert a six-digit palette color to an ANSI true-color foreground."""
    channels = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    return "38;2;" + ";".join(str(channel) for channel in channels)
