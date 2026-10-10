"""Human-readable diff gutters preserve source coordinates and literal text."""

from prompt_toolkit.formatted_text import fragment_list_to_text

from slipagent.viewers import DiffViewControl


def test_diff_coordinates_across_hunks_and_files():
    control = DiffViewControl("command\n\n--- a\n+++ b\n@@ -12,2 +20,3 @@\n keep\n-old\n+new\n+extra\n\\ No newline at end of file\n--- c\n+++ d\n@@ -0,0 +1 @@\n+first")
    try:
        # Random access builds the index without depending on render order.
        assert fragment_list_to_text(control.get_line(13)).startswith("            1 │ +first")
        assert fragment_list_to_text(control.get_line(5)) == "    12     20 │  keep"
        assert fragment_list_to_text(control.get_line(6)) == "    13        │ -old"
        assert fragment_list_to_text(control.get_line(7)) == "           21 │ +new"
        assert fragment_list_to_text(control.get_line(0)) == "command"
        assert fragment_list_to_text(control.get_line(1)) == ""
    finally:
        control.close()


def test_long_diff_line_continuations_do_not_consume_source_lines():
    control = DiffViewControl("@@ -1 +1,2 @@\n+" + "x" * 9000 + "\n+next\n")
    try:
        assert fragment_list_to_text(control.get_line(4)).startswith("            2 │ +next")
    finally:
        control.close()


def test_header_looking_source_lines_keep_addition_and_deletion_colors():
    control = DiffViewControl("@@ -1 +1 @@\n--- old text\n+++ new text")
    try:
        assert control.get_line(1)[1][0] == "class:diff-removed"
        assert control.get_line(2)[1][0] == "class:diff-added"
    finally:
        control.close()
