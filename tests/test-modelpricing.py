"""Free catalog ordering, labels, and menu presentation."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input

from slipagent.cli import _models_command, Style
from slipagent.modelpricing import free_label, price_label
from slipagent.terminal import TerminalUI
from slipagent.types import ModelInfo


@pytest.mark.parametrize("pricing,label", [({}, "-"), ({"prompt": "unknown"}, "-"),
    ({"prompt": 0, "completion": 0}, "free"),
    ({"free_tier": True}, "free tier"),
    ({"prompt": ".000001", "completion": ".000002"}, "$1.00/$2.00")])
def test_price_labels(pricing, label):
    assert price_label(pricing) == label


@pytest.mark.parametrize("query", ["", "free"])
async def test_free_models_first_with_matching_legend(query):
    choose = AsyncMock(return_value=None)
    models = [ModelInfo("a-paid", pricing={"prompt": 1, "completion": 1}),
              ModelInfo("b-unknown"), ModelInfo("z-free", pricing={"prompt": 0, "completion": 0}),
              ModelInfo("y-tier", pricing={"free_tier": True})]
    session = SimpleNamespace(catalog=AsyncMock(return_value=models), extensions={},
                              renderer=SimpleNamespace(terminal=SimpleNamespace(choose=choose)),
                              agent=SimpleNamespace(model="z-free"))
    await _models_command(session, query, Style(False), io.StringIO())
    options = choose.call_args.args[1]
    assert [value.split("::")[-1] for value, label in options][:2] == ["y-tier", "z-free"]
    assert len(options) == (2 if query else 4)
    styles = choose.call_args.kwargs["styles"]
    assert set(styles) == {m.selector for m in models if free_label(m.pricing)}
    assert choose.call_args.kwargs["legend"] == [("class:menu-free", "   "), ("", " = free")]


async def test_menu_legend_alignment_and_cleanup(tmp_path):
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", io.StringIO(), input=pipe, project_directory=tmp_path)
        choice = asyncio.create_task(ui.choose("Models", [("free", "Free model"), ("paid", "Paid model")],
                                              styles={"free": "class:menu-free"},
                                              legend=[("class:menu-free", "   "), ("", " = free")]))
        try:
            await asyncio.sleep(0)
            heading = fragment_list_to_text(ui._menu_heading())
            assert heading.endswith("   = free")
            assert len(heading) == ui.output.get_size().columns
            assert "class:menu-free" in ui._choice_panel.item_style(0)
            assert "class:menu-free" not in ui._choice_panel.item_style(1)
            ui._finish_menu(None)
            await choice
            assert ui.__dict__["_menu_legend"] == []
            assert ui._choice_panel.option_styles == {}
        finally:
            choice.cancel()
            await asyncio.gather(choice, return_exceptions=True)
            ui.close()
