"""Model search tables preserve columns and full identifiers at terminal widths."""

import io
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slipagent.cli import Renderer, Style, ModelPicker
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
    await ModelPicker(session, Style(False), out).browse("nemotron")
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
    await ModelPicker(session, Style(False), out).menu("nemotron")
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
    await ModelPicker(session, Style(False), out).browse(argument)
    output = out.getvalue()
    assert all(model.id in output for model in models)
    assert "75 shown of 75" in output
    assert "narrow with a filter" not in output
    assert output.index(models[0].id) < output.index(models[-1].id)


@pytest.mark.parametrize("selection", [None, "openrouter::test/free"])
async def test_free_model_selector_filters_and_switches_only_on_selection(monkeypatch, selection):
    from slipagent import cli
    models = [ModelInfo("test/paid", pricing={"prompt": "0.01", "completion": "0.01"}),
              ModelInfo("test/free", pricing={"prompt": "0", "completion": "0"})]
    catalog = AsyncMock(return_value=models)
    choose = AsyncMock(return_value=selection)
    renderer = Renderer(Style(False), io.StringIO(), False)
    renderer.terminal = SimpleNamespace(choose=choose)
    session = SimpleNamespace(renderer=renderer, catalog=catalog, agent=SimpleNamespace(model="test/free"))
    switch = AsyncMock()
    monkeypatch.setattr(cli.ModelPicker, "select", switch)
    await ModelPicker(session, Style(False), io.StringIO()).browse("free")
    title, options = choose.call_args.args
    assert title == "Select Model"
    assert [value for value, label in options] == ["openrouter::test/free"]
    assert "(current)" in options[0][1]
    if selection is None:
        switch.assert_not_awaited()
    else:
        assert switch.call_args.args[0] == selection


def model_session(choose, routing=None):
    return SimpleNamespace(renderer=SimpleNamespace(terminal=SimpleNamespace(choose=choose)),
                           client=SimpleNamespace(provider="openrouter"), agent=SimpleNamespace(model="test/current"),
                           model_routing=lambda: routing or SimpleNamespace(enabled=True))


async def test_model_without_arguments_lists_every_subcommand(monkeypatch):
    from slipagent import cli
    browse = AsyncMock()
    monkeypatch.setattr(cli.ModelPicker, "browse", browse)
    choose = AsyncMock(return_value="free")
    await ModelPicker(model_session(choose), Style(False), io.StringIO()).menu("")
    title, options = choose.call_args.args
    assert title == "Model: openrouter::test/current (Optimal on)"
    assert [label.split("]")[0] + "]" for _, label in options] == [
        "[model]", "[provider]", "[free]", "[optimal]", "[next]", "[reset]"]
    assert browse.call_args.args[0] == "free"


@pytest.mark.parametrize("argument, provider", [("nvidia", "nvidia"), ("provider", "kilo")])
async def test_model_provider_lists_that_providers_models(monkeypatch, argument, provider):
    from slipagent import cli
    browse = AsyncMock()
    monkeypatch.setattr(cli.ModelPicker, "browse", browse)
    choose = AsyncMock(return_value="kilo")
    await ModelPicker(model_session(choose), Style(False), io.StringIO()).menu(argument)
    assert browse.call_args.args[0] == "" and browse.call_args.kwargs["provider"] == provider


@pytest.mark.parametrize("argument", ["next", "free nemotron", "next ultra"])
async def test_single_words_are_subcommands_and_several_words_search(monkeypatch, argument):
    from slipagent import cli
    routing = AsyncMock()
    lookup = AsyncMock()
    monkeypatch.setattr(cli.ModelPicker, "routing_action", routing)
    monkeypatch.setattr(cli.ModelPicker, "lookup", lookup)
    await ModelPicker(model_session(AsyncMock()), Style(False), io.StringIO()).menu(argument)
    if argument == "next":
        assert routing.call_args.args[0] == "next" and not lookup.await_count
    else:
        assert lookup.call_args.args[0] == argument and not routing.await_count


async def test_provider_listing_shows_free_models_first_in_the_free_style():
    models = [ModelInfo("paid/model", provider="nvidia", pricing={"prompt": "1", "completion": "1"}),
              ModelInfo("free/model", provider="nvidia", pricing={"prompt": "0", "completion": "0"}),
              ModelInfo("other/model", provider="kilo", pricing={"prompt": "0", "completion": "0"})]
    choose = AsyncMock(return_value=None)
    renderer = Renderer(Style(False), io.StringIO(), False)
    renderer.terminal = SimpleNamespace(choose=choose)
    session = SimpleNamespace(renderer=renderer, catalog=AsyncMock(return_value=models), agent=SimpleNamespace(model="x"))
    await ModelPicker(session, Style(False), io.StringIO()).browse("", provider="nvidia")
    _, options = choose.call_args.args
    assert [value for value, _ in options] == ["nvidia::free/model", "nvidia::paid/model"]
    assert choose.call_args.kwargs["styles"] == {"nvidia::free/model": "class:menu-free"}


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
    await ModelPicker(session, Style(False), out).browse(query)
    if interactive:
        assert [value for value, label in choose.call_args.args[1]] == [models[0].selector]
    else:
        assert "1 shown of 3" in out.getvalue()
        assert models[0].id in out.getvalue()
        assert models[1].id not in out.getvalue()
        assert models[2].id not in out.getvalue()


@pytest.mark.parametrize("query,provider", [("google", "google"), ("GOOGLE", "google"),
    ("openrouter", "openrouter"), ("nvidia", "nvidia"),
    ("kilo", "kilo"), ("mistral", "mistral")])
async def test_provider_name_matches_all_its_models(query, provider):
    models = [ModelInfo("first", provider=provider), ModelInfo("second", provider=provider),
              ModelInfo("unrelated", provider="other")]
    choose = AsyncMock(return_value=None)
    session = SimpleNamespace(renderer=SimpleNamespace(terminal=SimpleNamespace(choose=choose)),
                              catalog=AsyncMock(return_value=models), agent=SimpleNamespace(model="first"))
    await ModelPicker(session, Style(False), io.StringIO()).browse(query)
    assert choose.await_count == 1
    assert [value for value, label in choose.call_args.args[1]] == [m.selector for m in models[:2]]


@pytest.mark.parametrize("query", ["google free", "FREE google", "google gemini free", "google flash", "google 1,000,000"])
async def test_search_matches_all_words_in_displayed_row(query):
    models = [ModelInfo("gemini-first", name="Flash", provider="google", context_length=1000000,
                        pricing={"free_tier": True}),
              ModelInfo("gemini-paid", provider="google", pricing={"prompt": 1, "completion": 1}),
              ModelInfo("other", provider="mistral", pricing={"prompt": 0, "completion": 0})]
    choose = AsyncMock(return_value=None)
    session = SimpleNamespace(renderer=SimpleNamespace(terminal=SimpleNamespace(choose=choose)),
                              catalog=AsyncMock(return_value=models), agent=SimpleNamespace(model="other"))
    await ModelPicker(session, Style(False), io.StringIO()).browse(query)
    options = choose.call_args.args[1]
    assert [value for value, label in options] == [models[0].selector]
    assert all(word in options[0][1].lower() for word in query.lower().split())
