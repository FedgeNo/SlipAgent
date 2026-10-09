"""Model search tables preserve columns and full identifiers at terminal widths."""

import io
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slipagent.cli import Renderer, Style, _model_command, _models_command
from slipagent.types import ModelInfo


@pytest.mark.parametrize("width", [20, 40, 80, 120])
async def test_models_search_renders_ascii_tables_that_fit(width, monkeypatch):
    model = ModelInfo("nvidia/nemotron-3-super-120b-a12b:free", context_length=262144)
    async def catalog(refresh=False):
        return [model, ModelInfo(model.id + "-long-identifier" * 10, context_length=1000000),
                ModelInfo("other/model", context_length=10000)]
    renderer = Renderer(Style(False), io.StringIO(), False)
    monkeypatch.setattr("slipagent.cli.shutil.get_terminal_size", lambda fallback: os.terminal_size((width, 24)))
    session = SimpleNamespace(renderer=renderer, catalog=catalog)
    out = io.StringIO()
    await _models_command(session, "nemotron", Style(False), out)
    lines = [line for line in out.getvalue().splitlines() if line.lstrip().startswith(("+", "|"))]
    assert lines
    assert all(len(line) <= width for line in lines)
    assert len({len(line) for line in lines}) == 1
    assert all(line.strip().isascii() for line in lines)
    assert "262,144" in out.getvalue()
    assert "free" in out.getvalue()
    if width >= 80:
        assert model.id in out.getvalue()


async def test_model_partial_match_suggestions_use_same_table():
    async def catalog(refresh=False):
        return [ModelInfo("nvidia/nemotron-3-super-120b-a12b:free", context_length=262144)]
    session = SimpleNamespace(renderer=Renderer(Style(False), io.StringIO(), False), catalog=catalog)
    out = io.StringIO()
    await _model_command(session, "nemotron", Style(False), out)
    assert "+" in out.getvalue() and "|" in out.getvalue()
    assert "nvidia/nemotron-3-super-120b-a12b:free" in out.getvalue()
    assert "262,144" in out.getvalue()


@pytest.mark.parametrize("argument", ["", "catalog/"])
async def test_models_displays_every_matching_catalog_entry(argument):
    models = [ModelInfo(f"catalog/model-{index:04d}", context_length=1000000) for index in range(75)]
    async def catalog():
        return list(reversed(models))
    session = SimpleNamespace(renderer=Renderer(Style(False), io.StringIO(), False), catalog=catalog)
    out = io.StringIO()
    await _models_command(session, argument, Style(False), out)
    output = out.getvalue()
    assert all(model.id in output for model in models)
    assert "75 shown of 75" in output
    assert "narrow with a filter" not in output
    assert output.index(models[0].id) < output.index(models[-1].id)


@pytest.mark.parametrize("selection", [None, "openrouter::test/free"])
async def test_free_model_selector_filters_and_switches_only_on_selection(monkeypatch, selection):
    from slipagent import cli
    models = [ModelInfo("test/paid", pricing={"prompt": "0.01", "completion": "0.01"}),
              ModelInfo("test/free")]
    catalog = AsyncMock(return_value=models)
    choose = AsyncMock(return_value=selection)
    renderer = Renderer(Style(False), io.StringIO(), False)
    renderer.terminal = SimpleNamespace(choose=choose)
    session = SimpleNamespace(renderer=renderer, catalog=catalog, agent=SimpleNamespace(model="test/free"))
    switch = AsyncMock()
    monkeypatch.setattr(cli, "_model_command", switch)
    await _models_command(session, "free", Style(False), io.StringIO())
    title, options = choose.call_args.args
    assert title == "Select Model"
    assert [value for value, label in options] == ["openrouter::test/free"]
    assert "(current)" in options[0][1]
    if selection is None:
        switch.assert_not_awaited()
    else:
        assert switch.call_args.args[1] == selection


async def test_model_without_slug_opens_selector(monkeypatch):
    from slipagent import cli
    browse = AsyncMock()
    monkeypatch.setattr(cli, "_models_command", browse)
    session = SimpleNamespace(renderer=SimpleNamespace(terminal=object()))
    out = io.StringIO()
    await _model_command(session, "", Style(False), out)
    assert browse.call_args.args[1] == ""


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("query", ["ultra nemotron", "NVIDIA   flagship", " nemotron\tultra  flagship "])
async def test_model_search_requires_every_word_across_fields(query, interactive):
    models = [
        ModelInfo("test/nemotron-3-ultra", name="Flagship", provider="nvidia"),
        ModelInfo("test/nemotron-super", name="Small", provider="nvidia"),
        ModelInfo("test/ultra", name="Flagship", provider="openrouter"),
    ]
    renderer = Renderer(Style(False), io.StringIO(), False)
    choose = AsyncMock(return_value=None)
    if interactive:
        renderer.terminal = SimpleNamespace(choose=choose)
    session = SimpleNamespace(renderer=renderer, catalog=AsyncMock(return_value=models),
                              agent=SimpleNamespace(model="test/nemotron-3-ultra"))
    out = io.StringIO()
    await _models_command(session, query, Style(False), out)
    if interactive:
        assert [value for value, label in choose.call_args.args[1]] == [models[0].selector]
    else:
        assert "1 shown of 3" in out.getvalue()
        assert models[0].id in out.getvalue()
        assert models[1].id not in out.getvalue()
        assert models[2].id not in out.getvalue()
