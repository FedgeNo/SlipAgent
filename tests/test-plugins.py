"""Plugin overrides: overlay, original chaining, fingerprints, isolation, and reload."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import types
from importlib import import_module
from pathlib import Path

import pytest
from offline.sitecustomize import GUARD_DIRECTORY

from slipagent import plugins

TARGET = '''
import io

STYLE = "target-style"


def helper():
    return "target-helper"


class Base:
    def greet(self):
        return "base"


class Widget(Base):
    """A target class."""

    def render(self, text):
        """Render text."""
        return f"<{text}>"

    def size(self):
        return 1

    def greet(self):
        return "widget"
'''


def component(source: str = TARGET, name: str = "widgets") -> types.ModuleType:
    module = types.ModuleType(f"fake.{name}")
    exec(source, module.__dict__)
    return module


def apply(module: types.ModuleType, files: dict[str, str], *, source: str = TARGET, core=frozenset(),
          approvals: plugins.Approvals | None = None) -> list[plugins.PluginReport]:
    sources = {f"{plugins.PLUGIN_KEY}/{name}": textwrap.dedent(text).encode() for name, text in files.items()}
    return plugins.overlay({"widgets": module}.get, lambda name: source.encode() if name == "widgets" else None,
                           sources, core, approvals)


def stamp(method: str) -> str:
    value = plugins.fingerprint(TARGET, "Widget", method)
    assert value is not None
    return value


def trust(approvals: plugins.Approvals, files: dict[str, str]) -> plugins.Approvals:
    """Approve fixture code independently of the override choices under test."""
    for name, source in files.items():
        approvals.decide_code(name, plugins.source_digest(textwrap.dedent(source).encode()), True)
    return approvals


def test_one_method_is_replaced_and_the_rest_stay_the_applications():
    module = component()
    reports = apply(module, {"quiet.py": f'''
        from slipagent.plugins import replaces, overrides

        @replaces("widgets.Widget")
        class Anything:
            @overrides("{stamp("render")}")
            def render(self, text):
                return f"[{{text}}]"

            def extra(self):
                return "added"
    '''})
    widget = module.Widget()
    assert widget.render("x") == "[x]"
    assert widget.size() == 1 and widget.extra() == "added"
    report = reports[0]
    assert report.enabled and not report.error and report.target == "widgets"
    assert [(o.member, o.status) for o in report.overrides] == [("render", "ok"), ("extra", "added")]
    assert not list(report.notices())


def test_original_filters_and_stacks_in_file_name_order():
    module = component()
    apply(module, {
        "20-outer.py": '''
            from slipagent.plugins import replaces, original

            @replaces("widgets.Widget")
            class Widget:
                def render(self, text):
                    return "outer(" + original(self, text) + ")"
        ''',
        "10-inner.py": '''
            from slipagent.plugins import replaces, original

            @replaces("widgets.Widget")
            class Widget:
                def render(self, text):
                    inner = lambda: original(self, text.upper())
                    return "inner(" + inner() + ")"
        ''',
    })
    assert module.Widget().render("x") == "outer(inner(<X>))"


PROPERTY_TARGET = '''
class Widget:
    def __init__(self):
        self._value = 10
        self.events = []

    @property
    def value(self):
        self.events.append("base get")
        return self._value

    @value.setter
    def value(self, value):
        self.events.append("base set")
        self._value = value

    @value.deleter
    def value(self):
        self.events.append("base delete")
        del self._value
'''

PROPERTY_PLUGIN = '''
from slipagent.plugins import replaces, original

@replaces("widgets.Widget")
class Widget:
    @property
    def value(self):
        self.events.append("plugin get")
        previous = lambda: original(self)
        return previous() + 1

    @value.setter
    def value(self, value):
        self.events.append("plugin set")
        original(self, value=value + 1)

    @value.deleter
    def value(self):
        self.events.append("plugin delete")
        original(self)
'''


@pytest.mark.parametrize("operation", ["get", "set", "delete"])
def test_original_calls_the_matching_property_accessor(operation):
    module = component(PROPERTY_TARGET)
    report = apply(module, {"property.py": PROPERTY_PLUGIN}, source=PROPERTY_TARGET)[0]
    assert report.enabled and not report.error
    widget = module.Widget()
    if operation == "get":
        assert widget.value == 11
    elif operation == "set":
        widget.value = 20
        assert widget._value == 21
    else:
        del widget.value
        assert not hasattr(widget, "_value")
    assert widget.events == [f"plugin {operation}", f"base {operation}"]


def test_original_returns_a_callable_property_value_without_calling_it():
    module = component(PROPERTY_TARGET)
    apply(module, {"property.py": '''
        from slipagent.plugins import replaces, original

        @replaces("widgets.Widget")
        class Widget:
            @property
            def value(self):
                return original(self)
    '''}, source=PROPERTY_TARGET)
    widget = module.Widget()
    called = []
    callback = lambda: called.append("called")
    widget._value = callback
    assert widget.value is callback
    assert not called and widget.events == ["base get"]


@pytest.mark.parametrize("operation", ["get", "set", "delete"])
def test_original_chains_each_property_accessor_through_stacked_plugins(operation):
    module = component(PROPERTY_TARGET)
    outer = PROPERTY_PLUGIN.replace('"plugin ', '"outer ').replace("+ 1", "+ 2")
    reports = apply(module, {"a-inner.py": PROPERTY_PLUGIN, "b-outer.py": outer}, source=PROPERTY_TARGET)
    assert all(report.enabled and not report.error for report in reports)
    widget = module.Widget()
    if operation == "get":
        assert widget.value == 13
    elif operation == "set":
        widget.value = 20
        assert widget._value == 23
    else:
        del widget.value
        assert not hasattr(widget, "_value")
    assert widget.events == [f"outer {operation}", f"plugin {operation}", f"base {operation}"]


@pytest.mark.parametrize("accessor, operation, label", [
    ("fget", "get", "getter"), ("fset", "set", "setter"), ("fdel", "delete", "deleter"),
])
def test_original_reports_when_the_replaced_property_lacks_that_accessor(accessor, operation, label):
    module = component(PROPERTY_TARGET)
    previous = module.Widget.value
    module.Widget.value = property(**{name: getattr(previous, name) for name in ("fget", "fset", "fdel")
                                     if name != accessor})
    apply(module, {"property.py": PROPERTY_PLUGIN}, source=PROPERTY_TARGET)
    widget = module.Widget()
    with pytest.raises(AttributeError, match=f"original property has no {label}"):
        if operation == "get":
            _ = widget.value
        elif operation == "set":
            widget.value = 20
        else:
            del widget.value
    assert widget.events == [f"plugin {operation}"]


def test_original_preserves_instance_static_and_class_method_binding():
    source = '''
class Widget:
    def instance(self, value):
        return self, value

    @staticmethod
    def static(value):
        return value * 2

    @classmethod
    def class_method(cls, value):
        return cls, value
'''
    module = component(source)
    apply(module, {"methods.py": '''
        from slipagent.plugins import replaces, original

        @replaces("widgets.Widget")
        class Widget:
            def instance(self, value):
                return original(self, value=value + 1)

            @staticmethod
            def static(value):
                return original(__class__, value=value) + 1

            @classmethod
            def class_method(cls, value):
                return original(cls, value=value + 2)
    '''}, source=source)
    widget = module.Widget()
    assert widget.instance(3) == (widget, 4)
    assert module.Widget.static(3) == widget.static(3) == 7
    assert module.Widget.class_method(3) == widget.class_method(3) == (module.Widget, 5)


def test_copied_code_resolves_names_from_the_target_module():
    module = component()
    apply(module, {"copy.py": '''
        from slipagent.plugins import replaces

        LOCAL = "plugin-local"

        @replaces("widgets.Widget")
        class Widget:
            def render(self, text):
                return (LOCAL, STYLE, helper(), io.StringIO().getvalue(), len(text),
                        [STYLE for _ in range(1)])
    '''})
    assert module.Widget().render("ab") == ("plugin-local", "target-style", "target-helper", "", 2, ["target-style"])


def test_zero_argument_super_reaches_the_targets_parent():
    module = component()
    apply(module, {"super.py": '''
        from slipagent.plugins import replaces

        @replaces("widgets.Widget")
        class Widget:
            def greet(self):
                return "plugin+" + super().greet()
    '''})
    assert module.Widget().greet() == "plugin+base"


def test_fingerprints_ignore_comments_formatting_and_docstrings_but_not_logic():
    edited = TARGET.replace('        """Render text."""\n', "").replace(
        'return f"<{text}>"', 'return f"<{text}>"   # same logic')
    assert plugins.fingerprint(edited, "Widget", "render") == stamp("render")
    changed = TARGET.replace('return f"<{text}>"', 'return f"<<{text}>>"')
    assert plugins.fingerprint(changed, "Widget", "render") != stamp("render")
    assert plugins.fingerprint(TARGET, "Widget", "missing") is None
    assert len(stamp("render")) == 12


def test_changed_and_unstamped_overrides_warn_but_still_apply():
    module = component()
    report = apply(module, {"old.py": '''
        from slipagent.plugins import replaces, overrides

        @replaces("widgets.Widget")
        class Widget:
            @overrides("000000000000")
            def render(self, text):
                return "old"

            def size(self):
                return 2
    '''})[0]
    widget = module.Widget()
    assert widget.render("x") == "old" and widget.size() == 2
    notices = list(report.notices())
    assert "old.py overrides widgets.Widget.render, which has changed since this override was written" in notices[0]
    assert f'add @overrides("{stamp("size")}")' in notices[1]


@pytest.mark.parametrize("files, reason", [
    ({"gone.py": '''
        from slipagent.plugins import replaces, overrides

        @replaces("widgets.Widget")
        class Widget:
            def render(self, text):
                return "applied first"

            @overrides("123456789abc")
            def removed(self):
                return 1
    '''}, "widgets.Widget.removed no longer exists"),
    ({"bases.py": '''
        from slipagent.plugins import replaces

        class Other:
            pass

        @replaces("widgets.Widget")
        class Widget(Other):
            def render(self, text):
                return "x"
    '''}, "may omit base classes"),
    ({"two.py": '''
        from slipagent.plugins import replaces

        @replaces("widgets.Widget")
        class Widget:
            pass

        @replaces("tools.Other")
        class Other:
            pass
    '''}, "one file may target one module"),
    ({"core.py": '''
        from slipagent.plugins import replaces

        @replaces("runtime.RuntimeFrame")
        class Frame:
            pass
    '''}, "stable frame"),
    ({"syntax.py": "def broken(:\n"}, "SyntaxError"),
    ({"none.py": "VALUE = 1\n"}, "no class is marked"),
    ({"missing.py": '''
        from slipagent.plugins import replaces

        @replaces("widgets.Missing")
        class Missing:
            pass
    '''}, "class widgets.Missing does not exist"),
])
def test_a_broken_plugin_is_disabled_without_touching_the_application(files, reason):
    module = component()
    before = dict(module.Widget.__dict__)
    report = apply(module, files, core=frozenset({"runtime"}))[0]
    assert not report.enabled and reason in report.error
    assert next(report.notices()).startswith(f"Plugin {next(iter(files))} disabled:")
    assert dict(module.Widget.__dict__) == before
    assert module.Widget().render("x") == "<x>"


def test_a_broken_plugin_does_not_stop_the_others():
    module = component()
    reports = apply(module, {
        "a-broken.py": "raise RuntimeError('boom')\n",
        "b-good.py": '''
            from slipagent.plugins import replaces

            @replaces("widgets.Widget")
            class Widget:
                def size(self):
                    return 5
        ''',
    })
    assert [report.enabled for report in reports] == [False, True]
    assert module.Widget().size() == 5


PICK = '''
    from slipagent.plugins import replaces

    @replaces("widgets.Widget")
    class Widget:
        def render(self, text):
            return "plugin"

        def size(self):
            return 9
'''


def test_saved_approvals_decide_each_override_and_undecided_ones_stay_off():
    module = component()
    approvals = trust(plugins.Approvals({"pick.py": {"Widget.size": True}}), {"pick.py": PICK})
    report = apply(module, {"pick.py": PICK}, approvals=approvals)[0]
    assert module.Widget().render("x") == "<x>" and module.Widget().size() == 9
    assert [(override.member, override.state) for override in report.overrides] == [
        ("render", "pending"), ("size", "on")]
    assert list(report.notices())[-1] == (
        "pick.py has 1 override awaiting approval; they stay off until approved with /plugins.")
    assert plugins.pending([report])

    module = component()
    approvals.decide("pick.py", "Widget.render", False)
    report = apply(module, {"pick.py": PICK}, approvals=approvals)[0]
    assert module.Widget().render("x") == "<x>" and report.overrides[0].state == "rejected"
    assert not any("/plugins" in text for text in report.notices()) and not plugins.pending([report])


def test_approvals_survive_a_round_trip_and_ignore_malformed_content():
    approvals = trust(plugins.Approvals(), {"a.py": PICK})
    approvals.decide("a.py", "Widget.render", True)
    approvals.choose("widgets.Widget.render", ["b.py", "a.py"], ["b.py", "a.py"])
    parsed = plugins.Approvals.parse(approvals.encode())
    assert parsed == approvals
    assert parsed.order("widgets.Widget.render", ["a.py", "b.py"]) == ["b.py", "a.py"]
    assert parsed.order("widgets.Widget.render", ["a.py", "b.py", "c.py"]) is None
    for content in (b"{not json", b"[]", b'{"approved": {"a.py": {"Widget.render": "yes"}}}'):
        assert plugins.Approvals.parse(content).decision("a.py", "Widget.render") is None


@pytest.mark.parametrize("old_decision", [None, True, False])
def test_unapproved_plugin_code_is_not_executed(tmp_path, old_decision):
    marker = tmp_path / "executed"
    source = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n' + textwrap.dedent(PICK)
    module = component()
    decisions = plugins.Approvals()
    if old_decision is not None:
        decisions.decide("pending.py", "Widget.size", old_decision)
    report = apply(module, {"pending.py": source}, approvals=decisions)[0]
    assert not marker.exists()
    assert module.Widget().size() == 1
    assert plugins.pending([report])


def test_code_approval_covers_the_exact_source_and_rejection_stays_inert(tmp_path):
    marker = tmp_path / "executed"
    source = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n' + textwrap.dedent(PICK)
    decisions = trust(plugins.Approvals({"code.py": {"Widget.size": True}}), {"code.py": source})
    module = component()
    assert apply(module, {"code.py": source}, approvals=decisions)[0].enabled
    assert marker.read_text() == "ran" and module.Widget().size() == 9
    marker.unlink()
    changed = source + "\n# changed source\n"
    report = apply(component(), {"code.py": changed}, approvals=decisions)[0]
    assert report.code_state == "pending" and not marker.exists()
    decisions.decide_code("code.py", report.digest, False)
    report = apply(component(), {"code.py": changed}, approvals=decisions)[0]
    assert report.code_state == "rejected" and not marker.exists()
    assert not plugins.pending([report])


@pytest.mark.parametrize("content", [
    b'{"approved":[1]}', b'{"conflicts":"invalid"}',
    b'{"conflicts":{"widgets.Widget.render":{"plugins":[1,"a.py"],"order":["a.py"]}}}',
    b'{"code":[]}', b'{"code":{"a.py":{"sha256":"invalid","approved":true}}}',
    b'{"approved":{"a.py":{"Widget.size":true}},"conflicts":null}',
])
def test_malformed_approval_fields_disable_decisions_and_report_the_problem(content):
    decisions = plugins.Approvals.parse(content)
    assert decisions.approved == {} and decisions.conflicts == {}
    assert decisions.error


def test_invalid_approval_file_reports_a_notice_without_loading_plugins(tmp_path):
    marker = tmp_path / "executed"
    source = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n'
    decisions = plugins.Approvals.parse(b'{"approved":[1]}')
    reports = apply(component(), {"code.py": source}, approvals=decisions)
    assert not marker.exists()
    assert any("Invalid .approvals.json" in notice for report in reports for notice in report.notices())


STACKABLE = {
    "a.py": '''
        from slipagent.plugins import replaces, original

        @replaces("widgets.Widget")
        class Widget:
            def render(self, text):
                return "a(" + original(self, text) + ")"
    ''',
    "b.py": '''
        from slipagent.plugins import replaces, original

        @replaces("widgets.Widget")
        class Widget:
            def render(self, text):
                helper = lambda: original(self, text)
                return "b(" + helper() + ")"
    ''',
}


def allowed_both() -> plugins.Approvals:
    return trust(plugins.Approvals({"a.py": {"Widget.render": True}, "b.py": {"Widget.render": True}}), STACKABLE)


def test_a_stacked_override_behind_one_without_original_is_reported_dead():
    module = component()
    reports = apply(module, {**STACKABLE, "c.py": PICK})
    assert module.Widget().render("x") == "plugin"
    assert [report.overrides[0].shadowed_by for report in reports] == ["c.py", "c.py", ""]
    assert "a.py's override of widgets.Widget.render never runs: c.py runs before it and does not call original()." \
        in list(reports[0].notices())
    module = component()
    reports = apply(module, {**STACKABLE, "0.py": PICK})
    assert module.Widget().render("x") == "b(a(plugin))"
    assert not any(override.shadowed_by for report in reports for override in report.overrides)


def test_an_unresolved_conflict_applies_neither_override_and_is_reported_once():
    module = component()
    reports = apply(module, STACKABLE, approvals=allowed_both())
    assert module.Widget().render("x") == "<x>"
    assert [override.state for report in reports for override in report.overrides] == ["conflict", "conflict"]
    assert [list(report.notices()) for report in reports] == [[
        "widgets.Widget.render is overridden by a.py, b.py; none of them apply until one is chosen with /plugins."], []]
    assert all(report.overrides[0].calls_original for report in reports)


@pytest.mark.parametrize(("order", "result", "states"), [
    (["a.py", "b.py"], "a(b(<x>))", ["on", "on"]),
    (["b.py", "a.py"], "b(a(<x>))", ["on", "on"]),
    (["b.py"], "b(<x>)", ["dropped", "on"]),
])
def test_conflict_choices_pick_one_plugin_or_a_stacking_order(order, result, states):
    module = component()
    approvals = allowed_both()
    approvals.choose("widgets.Widget.render", ["a.py", "b.py"], order)
    reports = apply(module, STACKABLE, approvals=approvals)
    assert module.Widget().render("x") == result
    assert [report.overrides[0].state for report in reports] == states
    assert not any("/plugins" in text for report in reports for text in report.notices())


def test_without_approvals_every_override_applies_stacked_in_file_name_order():
    module = component()
    apply(module, STACKABLE)
    assert module.Widget().render("x") == "b(a(<x>))"


class ScriptedChooser:
    """Answers `TerminalUI.choose` from a script, recording each title and its option keys."""

    def __init__(self, answers: list[str | None]) -> None:
        self.answers = answers
        self.asked: list[tuple[str, list[str]]] = []

    async def choose(self, title, options, **_):
        self.asked.append((title, [key for key, _ in options]))
        return self.answers.pop(0)


def menu_for(answers: list[str | None], monkeypatch) -> tuple[object, ScriptedChooser, list[object]]:
    from slipagent import cli

    chooser = ScriptedChooser(answers)
    applied: list[object] = []
    monkeypatch.setattr(plugins, "apply_to_live", lambda core: applied.append(core) or [])
    session = types.SimpleNamespace(renderer=types.SimpleNamespace(terminal=chooser), reloader=None)
    out = __import__("io").StringIO()
    return cli.PluginMenu(session, cli.Style(False), out), chooser, applied


@pytest.mark.parametrize("answer", [None, "reject", "approve"])
async def test_menu_gets_code_approval_before_execution_and_then_asks_about_methods(tmp_path, monkeypatch, answer):
    marker = tmp_path / "executed"
    source = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n' + textwrap.dedent(PICK)
    files = {"code.py": source}
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=plugins.Approvals()))
    menu, chooser, _ = menu_for([answer, "all"] if answer == "approve" else [answer], monkeypatch)
    module = component()

    def apply_saved(core):
        plugins.reports = apply(module, files, approvals=plugins.load_approvals())
        return plugins.reports

    monkeypatch.setattr(plugins, "apply_to_live", apply_saved)
    assert not marker.exists()
    await menu.offer()
    assert chooser.asked[0] == ("Python Code: code.py", ["approve", "reject"])
    assert marker.exists() == (answer == "approve")
    assert module.Widget().size() == (9 if answer == "approve" else 1)
    assert len(chooser.asked) == (2 if answer == "approve" else 1)
    assert plugins.pending(plugins.reports) == (answer is None)


async def test_code_changed_while_approval_menu_is_open_stays_unloaded(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    files = {"code.py": textwrap.dedent(PICK)}
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=plugins.Approvals()))
    menu, chooser, _ = menu_for(["approve"], monkeypatch)
    choose = chooser.choose

    async def changed_source(*args, **kwargs):
        files["code.py"] += f'\nfrom pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n'
        return await choose(*args, **kwargs)

    def apply_saved(core):
        plugins.reports = apply(component(), files, approvals=plugins.load_approvals())
        return plugins.reports

    monkeypatch.setattr(chooser, "choose", changed_source)
    monkeypatch.setattr(plugins, "apply_to_live", apply_saved)
    await menu.offer()
    assert not marker.exists() and plugins.pending(plugins.reports)


async def test_the_menu_approves_new_plugins_then_asks_about_their_conflict(monkeypatch):
    decisions = trust(plugins.Approvals(), STACKABLE)
    plugins.save_approvals(decisions)
    monkeypatch.setattr(plugins, "reports", apply(component(), STACKABLE, approvals=decisions))
    menu, chooser, applied = menu_for(["all", "all", "b.py/a.py"], monkeypatch)
    await menu.offer()
    assert chooser.asked == [
        ("Plugin a.py → Widget.render", ["all", "choose", "none"]),
        ("Plugin b.py → Widget.render", ["all", "choose", "none"]),
        ("widgets.Widget.render is overridden by a.py, b.py", ["a.py", "b.py", "a.py/b.py", "b.py/a.py"]),
    ]
    saved = plugins.load_approvals()
    assert saved.approved == allowed_both().approved
    assert saved.order("widgets.Widget.render", ["a.py", "b.py"]) == ["b.py", "a.py"]
    assert len(applied) == 1

    monkeypatch.setattr(plugins, "reports", apply(component(), STACKABLE, approvals=saved))
    menu, chooser, applied = menu_for([], monkeypatch)
    await menu.offer()
    assert chooser.asked == [] and applied == []


async def test_choosing_overrides_one_by_one_and_stacking_only_through_original(monkeypatch):
    files = {**STACKABLE, "c.py": PICK}
    decisions = trust(allowed_both(), {"c.py": PICK})
    plugins.save_approvals(decisions)
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=decisions))
    menu, chooser, applied = menu_for(["choose", "allow", "reject", None], monkeypatch)
    await menu.offer()
    assert chooser.asked[0] == ("Plugin c.py → Widget.render, Widget.size", ["all", "choose", "none"])
    assert [title for title, _ in chooser.asked[1:3]] == ["c.py → Widget.render", "c.py → Widget.size"]
    # c.py replaces render without calling original(), so it can only run last in a stack.
    title, keys = chooser.asked[3]
    assert title == "widgets.Widget.render is overridden by a.py, b.py, c.py"
    assert keys[:3] == ["a.py", "b.py", "c.py"]
    assert {tuple(key.split("/")) for key in keys[3:]} == {("a.py", "b.py", "c.py"), ("b.py", "a.py", "c.py")}
    saved = plugins.load_approvals()
    assert saved.approved["c.py"] == {"Widget.render": True, "Widget.size": False}
    assert saved.order("widgets.Widget.render", ["a.py", "b.py", "c.py"]) is None
    assert len(applied) == 1


async def test_rejecting_new_overrides_keeps_existing_approved_methods(monkeypatch):
    decisions = trust(plugins.Approvals({"pick.py": {"Widget.size": True}}), {"pick.py": PICK})
    report = apply(component(), {"pick.py": PICK}, approvals=decisions)[0]
    menu, chooser, _ = menu_for(["none"], monkeypatch)
    await menu.approve(chooser, decisions, report, [report.overrides[0]])
    assert decisions.decision("pick.py", "Widget.render") is False
    assert decisions.code_decision("pick.py", report.digest) is True
    module = component()
    apply(module, {"pick.py": PICK}, approvals=decisions)
    assert module.Widget().size() == 9


def test_original_outside_an_override_explains_itself():
    with pytest.raises(RuntimeError, match="original"):
        plugins.original(object())


def test_plugin_files_skip_disabled_names_and_honour_the_switches(monkeypatch):
    directory = plugins.plugin_directory()
    directory.mkdir(parents=True)
    for name in ("20-b.py", "10-a.py", "_off.py", ".hidden.py", "notes.txt"):
        (directory / name).write_text("")
    assert [path.name for path in plugins.plugin_files()] == ["10-a.py", "20-b.py"]
    assert set(plugins.plugin_sources()) == {f"{plugins.PLUGIN_KEY}/10-a.py", f"{plugins.PLUGIN_KEY}/20-b.py"}
    plugins.save_approvals(plugins.Approvals({"10-a.py": {"Widget.render": True}}))
    assert f"{plugins.PLUGIN_KEY}/{plugins.APPROVALS_NAME}" in plugins.plugin_stamps()
    files, approvals = plugins.split_sources(plugins.plugin_sources())
    assert set(files) == {f"{plugins.PLUGIN_KEY}/10-a.py", f"{plugins.PLUGIN_KEY}/20-b.py"}
    assert approvals is not None and approvals.decision("10-a.py", "Widget.render") is True
    monkeypatch.setattr(plugins, "all_plugins", True)
    assert plugins.split_sources(plugins.plugin_sources())[1] is None
    monkeypatch.setenv(plugins.NO_PLUGINS_ENV, "1")
    assert plugins.plugin_files() == []
    monkeypatch.delenv(plugins.NO_PLUGINS_ENV)
    monkeypatch.setattr(plugins, "disabled", True)
    assert plugins.plugin_files() == []


SETUP = import_module("test-runtime").SETUP


@pytest.fixture
def run_copy(tmp_path: Path, reload_source):
    package_root = tmp_path / "source"
    shutil.copytree(reload_source, package_root)
    project = tmp_path / "project"
    project.mkdir()

    def run(body: str, plugin_files: dict[str, str] | None = None) -> None:
        directory = Path(os.environ["SLIPAGENT_STATE_DIR"]) / "plugins"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / plugins.APPROVALS_NAME).write_bytes(APPROVED.encode())
        for name, text in (plugin_files or {}).items():
            (directory / name).write_text(textwrap.dedent(text))
        script = SETUP.replace("__BODY__", textwrap.indent(
            "from slipagent import plugins\nplugins_dir = Path(os.environ['SLIPAGENT_STATE_DIR']) / 'plugins'\n"
            + textwrap.dedent(body).strip(), "        "))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(package_root)])}
        result = subprocess.run([sys.executable, "-c", script], cwd=project, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr

    return run


APPROVED = plugins.Approvals({"loud.py": {"Renderer.emit": True}, "contract.py": {"Renderer.emit": True}})
RENDERER_PLUGIN = '''
from slipagent.plugins import replaces, original

@replaces("cli.Renderer")
class Renderer:
    def emit(self, text):
        return original(self, "PLUGIN " + str(text))
'''
trust(APPROVED, {"loud.py": RENDERER_PLUGIN})


def test_saving_a_plugin_reloads_and_applies_it_to_live_classes(run_copy):
    run_copy(f'''
        renderer.emit('before')
        assert 'PLUGIN' not in sink.getvalue()
        (plugins_dir / 'loud.py').write_text({RENDERER_PLUGIN!r})
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        renderer.emit('after')
        assert 'PLUGIN after' in sink.getvalue(), sink.getvalue()
        (plugins_dir / 'loud.py').rename(plugins_dir / '_loud.py')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        renderer.emit('removed')
        assert 'PLUGIN removed' not in sink.getvalue()
    ''')


def test_a_broken_plugin_is_reported_once_and_the_reload_still_applies(run_copy):
    run_copy('''
        (plugins_dir / 'broken.py').write_text('raise ValueError("bad plugin")\\n')
        decisions = plugins.load_approvals()
        decisions.decide_code('broken.py', plugins.source_digest((plugins_dir / 'broken.py').read_bytes()), True)
        plugins.save_approvals(decisions)
        edit('statusline.py', 'NOTICE_SECONDS = 30', 'NOTICE_SECONDS = 31')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert sink.getvalue().count('Plugin broken.py disabled: ValueError: bad plugin') == 1
        edit('statusline.py', 'NOTICE_SECONDS = 31', 'NOTICE_SECONDS = 32')
        await frame.checkpoint()
        assert frame.generation == 2
        assert sink.getvalue().count('Plugin broken.py disabled') == 1
    ''')


def test_a_plugin_that_breaks_a_component_contract_is_rejected_like_an_edit(run_copy):
    run_copy('''
        (plugins_dir / 'contract.py').write_text(
            'from slipagent.plugins import replaces\\n'
            '@replaces("cli.Renderer")\\n'
            'class Renderer:\\n'
            '    emit = None\\n')
        decisions = plugins.load_approvals()
        decisions.decide_code('contract.py', plugins.source_digest((plugins_dir / 'contract.py').read_bytes()), True)
        plugins.save_approvals(decisions)
        await frame.checkpoint()
        assert frame.generation == 0
        assert 'Missing component API in cli.Renderer' in sink.getvalue(), sink.getvalue()
        renderer.emit('still works')
        assert 'still works' in sink.getvalue()
    ''')


def test_startup_applies_plugins_unless_disabled(tmp_path, reload_source):
    package_root = tmp_path / "source"
    shutil.copytree(reload_source, package_root)
    directory = Path(os.environ["SLIPAGENT_STATE_DIR"]) / "plugins"
    directory.mkdir(parents=True)
    (directory / "loud.py").write_text(RENDERER_PLUGIN)
    (directory / plugins.APPROVALS_NAME).write_bytes(APPROVED.encode())
    script = textwrap.dedent('''
        import io, sys
        import slipagent.cli as cli
        from slipagent import plugins
        from slipagent.runtime import CORE_MODULES
        plugins.disabled = "--no-plugins" in sys.argv
        reports = plugins.apply_to_live(CORE_MODULES)
        sink = io.StringIO()
        cli.Renderer(cli.Style(False), sink, False).emit("hello")
        print(sink.getvalue().strip(), [report.enabled for report in reports])
    ''')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(package_root)])}
    for extra, expected in (([], "PLUGIN hello [True]"), (["--no-plugins"], "hello []")):
        result = subprocess.run([sys.executable, "-c", script, *extra], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected


def test_reapplying_to_live_classes_follows_changed_approvals(tmp_path, reload_source):
    package_root = tmp_path / "source"
    shutil.copytree(reload_source, package_root)
    directory = Path(os.environ["SLIPAGENT_STATE_DIR"]) / "plugins"
    directory.mkdir(parents=True)
    (directory / "loud.py").write_text(RENDERER_PLUGIN)
    script = textwrap.dedent('''
        import io
        import slipagent.cli as cli
        from slipagent import plugins
        from slipagent.runtime import CORE_MODULES

        def emitted():
            sink = io.StringIO()
            cli.Renderer(cli.Style(False), sink, False).emit("hello")
            return sink.getvalue().strip()

        approvals = plugins.Approvals()
        approvals.decide_code("loud.py", plugins.source_digest((plugins.plugin_directory() / "loud.py").read_bytes()), True)
        plugins.save_approvals(approvals)
        for decision in (None, True, False, True):
            if decision is not None:
                approvals.decide("loud.py", "Renderer.emit", decision)
                plugins.save_approvals(approvals)
            plugins.apply_to_live(CORE_MODULES)
            print(emitted(), plugins.reports[0].overrides[0].state)
    ''')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(package_root)])}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "hello pending", "PLUGIN hello on", "hello rejected", "PLUGIN hello on"]


@pytest.fixture
def workshop(tmp_path, monkeypatch):
    """Plugin tools over a fake package whose `widgets` module is TARGET."""
    from slipagent.plugintools import PluginTools

    package = tmp_path / "package"
    package.mkdir()
    (package / "widgets.py").write_text(TARGET)
    (package / "config.py").write_text("class Config:\n    pass\n")
    monkeypatch.setattr(plugins, "PACKAGE_ROOT", package)
    return PluginTools()


def load(path: Path, module: types.ModuleType, approvals: plugins.Approvals | None = None,
         source: str | None = None) -> plugins.PluginReport:
    application = source if source is not None else (plugins.PACKAGE_ROOT / "widgets.py").read_text()
    return plugins.overlay({"widgets": module}.get, lambda name: application.encode(),
                           {plugins.source_key(path): path.read_bytes()}, frozenset({"config"}), approvals)[0]


def test_copying_a_class_writes_stamped_methods_that_load_cleanly(workshop):
    assert workshop.targets() == ["widgets.Base", "widgets.Widget"]
    path, members = workshop.copy("widgets.Widget")
    assert path.name == "widgets-widget.py"
    assert members == ["Widget.render", "Widget.size", "Widget.greet"]
    module = component()
    report = load(path, module)
    assert not report.error and [override.status for override in report.overrides] == ["ok", "ok", "ok"]
    assert module.Widget().render("x") == "<x>" and module.Widget().greet() == "widget"
    assert workshop.load_baseline(path)["Widget.size"] == "def size(self):\n    return 1"


def test_copying_methods_one_at_a_time_appends_to_the_same_stand_in(workshop):
    workshop.copy("widgets.Widget.render", "mine")
    path, _ = workshop.copy("widgets.Widget.size", "mine.py")
    workshop.copy("widgets.Base.greet", "mine")
    source = path.read_text()
    assert source.count('@replaces("widgets.Widget")') == 1 and source.count('@replaces("widgets.Base")') == 1
    module = component()
    report = load(path, module)
    assert [(override.target, override.member, override.status) for override in report.overrides] == [
        ("widgets.Widget", "render", "ok"), ("widgets.Widget", "size", "ok"), ("widgets.Base", "greet", "ok")]
    from slipagent.plugintools import PluginToolError
    with pytest.raises(PluginToolError, match="already defines Widget.render"):
        workshop.copy("widgets.Widget.render", "mine")
    with pytest.raises(PluginToolError, match="stable frame"):
        workshop.copy("config.Config", "other")
    with pytest.raises(PluginToolError, match="names no module.Class"):
        workshop.copy("widgets.Missing")
    with pytest.raises(PluginToolError, match="does not exist"):
        workshop.copy("widgets.Widget.missing")


def test_diff_shows_the_upstream_change_and_restamp_records_it(workshop):
    path, _ = workshop.copy("widgets.Widget.render", "mine")
    assert workshop.diff("mine") == []
    changed = TARGET.replace('return f"<{text}>"', 'return f"<<{text}>>"')
    (plugins.PACKAGE_ROOT / "widgets.py").write_text(changed)
    assert load(path, component(changed)).overrides[0].status == "changed"
    diff = workshop.diff("mine")
    assert '-    return f"<{text}>"' in diff and '+    return f"<<{text}>>"' in diff
    assert workshop.diff() == diff
    assert workshop.restamp("mine") == ["Widget.render"]
    assert workshop.diff("mine") == [] and workshop.restamp("mine") == []
    assert load(path, component(changed)).overrides[0].status == "ok"


def test_restamp_stamps_unstamped_overrides_and_adds_the_import(workshop):
    path = workshop.resolve_file("bare")
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent('''
        from slipagent.plugins import replaces

        @replaces("widgets.Widget")
        class Widget:
            @staticmethod
            def size():
                return 2

            def helper(self):
                return "mine"
    '''))
    assert load(path, component()).overrides[0].status == "unstamped"
    from slipagent.plugintools import PluginToolError
    with pytest.raises(PluginToolError, match="no override of an application method named helper"):
        workshop.restamp("bare", "helper")
    assert workshop.restamp("bare", "size") == ["Widget.size"]
    module = component()
    report = load(path, module)
    assert [override.status for override in report.overrides] == ["ok", "added"]
    assert module.Widget().size() == 2


async def test_plugin_copy_approves_what_it_wrote_and_the_listing_shows_each_state(workshop, monkeypatch):
    menu, chooser, applied = menu_for(["widgets.Widget", "size"], monkeypatch)
    await menu.command("copy")
    assert [title for title, _ in chooser.asked] == ["Copy into a plugin", "Copy from widgets.Widget"]
    assert plugins.load_approvals().approved == {"widgets-widget.py": {"Widget.size": True}}
    path = workshop.resolve_file("widgets-widget")
    assert plugins.load_approvals().code_decision(path.name, plugins.source_digest(path.read_bytes())) is True
    assert len(applied) == 1 and "Wrote Widget.size" in menu.out.getvalue()

    files = {**STACKABLE, "c.py": PICK}
    approvals = trust(allowed_both(), {"c.py": PICK})
    approvals.decide("c.py", "Widget.size", False)
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=approvals))
    menu, _, _ = menu_for([], monkeypatch)
    menu.listing()
    assert menu.out.getvalue().splitlines()[1:] == [
        "  a.py → widgets", "    Widget.render: conflict unresolved, unstamped",
        "  b.py → widgets", "    Widget.render: conflict unresolved, unstamped",
        "  c.py → widgets", "    Widget.render: awaiting approval, unstamped", "    Widget.size: rejected, unstamped",
    ]


async def test_plugin_copy_does_not_approve_preexisting_code(workshop, tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    path = workshop.resolve_file("existing")
    path.parent.mkdir(parents=True)
    path.write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n')
    menu, _, _ = menu_for([], monkeypatch)
    await menu.command("copy widgets.Widget.size existing")
    decisions = plugins.load_approvals()
    report = load(path, component(), decisions)
    assert decisions.decision(path.name, "Widget.size") is True
    assert report.code_state == "pending" and not marker.exists()
