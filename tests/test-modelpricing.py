"""Free catalog ordering, labels, and menu presentation."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input

from slipagent.cli import ModelPicker, Style
from slipagent.modelpricing import free_label, price_label
from slipagent.terminal import TerminalUI
from slipagent.types import ModelInfo, Usage


@pytest.mark.parametrize("pricing,label", [({}, "-"), ({"prompt": "unknown"}, "-"),
    ({"prompt": 0, "completion": 0}, "free"),
    ({"free_tier": True}, "free tier"),
    ({"prompt": ".000001", "completion": ".000002"}, "$1.00/$2.00")])
def test_price_labels(pricing, label):
    assert price_label(pricing) == label


@pytest.mark.parametrize("known_cost", [0, .02])
@pytest.mark.parametrize("unknown_first", [False, True])
def test_session_cost_is_hidden_when_any_token_usage_is_unpriced(known_cost, unknown_first):
    unknown = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    known = Usage(prompt_tokens=2, total_tokens=2, cost=known_cost)
    total = unknown + known if unknown_first else known + unknown
    total = total + Usage(cost=0)
    assert total.cost is None
    assert total.summary() == "in=12 out=5"


def test_empty_usage_does_not_hide_a_reported_zero_cost():
    usage = Usage() + Usage(prompt_tokens=10, cost=0)
    assert usage.cost == 0
    assert "cost=$0.000000" in usage.summary()


@pytest.mark.parametrize("cost", [None, .02])
def test_free_compaction_does_not_contribute_to_session_accounting(cost):
    from slipagent.agent import Agent
    from slipagent.tools.base import ToolRegistry
    from test_agent import StubClient

    agent = Agent(StubClient([]), ToolRegistry(), "test")
    working = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=cost)
    agent.usage = working
    agent._compactor().on_usage(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0))
    assert agent.usage == working
    assert agent.usage.summary() == working.summary()


@pytest.mark.parametrize("width", [20, 80])
def test_model_prices_are_hidden_when_unavailable(width):
    session = SimpleNamespace(renderer=SimpleNamespace(terminal=SimpleNamespace(
        output=SimpleNamespace(get_size=lambda: SimpleNamespace(columns=width)))))
    picker = ModelPicker(session, Style(False), io.StringIO())
    model = ModelInfo("claude-test", provider="anthropic", context_length=200000,
                      pricing={"prompt": "unknown", "completion": "unknown"})
    table = picker.table([model])
    assert "$/Mtok" not in table
    assert "$" not in picker.label(model)
    assert "  -  " not in picker.label(model)
    assert picker.describe(model).endswith("200k")
    assert all(len(line) <= width for line in table.splitlines())


def test_mixed_catalog_preserves_available_prices():
    session = SimpleNamespace(renderer=SimpleNamespace(terminal=None))
    picker = ModelPicker(session, Style(False), io.StringIO())
    models = [ModelInfo("unknown"), ModelInfo("paid", pricing={"prompt": ".000001", "completion": ".000002"})]
    table = picker.table(models)
    assert "In/Out $/Mtok" in table
    assert "$1.00/$2.00" in table
    assert "$1.00/$2.00/Mtok" in picker.label(models[1])


@pytest.mark.parametrize("query", ["", "free"])
async def test_free_models_first_with_matching_legend(query):
    choose = AsyncMock(return_value=None)
    models = [ModelInfo("a-paid", pricing={"prompt": 1, "completion": 1}),
              ModelInfo("b-unknown"), ModelInfo("z-free", pricing={"prompt": 0, "completion": 0}),
              ModelInfo("y-tier", pricing={"free_tier": True})]
    session = SimpleNamespace(catalog=AsyncMock(return_value=models), extensions={},
                              renderer=SimpleNamespace(terminal=SimpleNamespace(choose=choose)),
                              agent=SimpleNamespace(model="z-free"))
    await ModelPicker(session, Style(False), io.StringIO()).browse(query)
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
