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
    approvals = plugins.Approvals({"pick.py": {"Widget.size": True}})
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
    approvals = plugins.Approvals()
    approvals.decide("a.py", "Widget.render", True)
    approvals.choose("widgets.Widget.render", ["b.py", "a.py"], ["b.py", "a.py"])
    parsed = plugins.Approvals.parse(approvals.encode())
    assert parsed == approvals
    assert parsed.order("widgets.Widget.render", ["a.py", "b.py"]) == ["b.py", "a.py"]
    assert parsed.order("widgets.Widget.render", ["a.py", "b.py", "c.py"]) is None
    for content in (b"{not json", b"[]", b'{"approved": {"a.py": {"Widget.render": "yes"}}}'):
        assert plugins.Approvals.parse(content).decision("a.py", "Widget.render") is None


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
    return plugins.Approvals({"a.py": {"Widget.render": True}, "b.py": {"Widget.render": True}})


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


async def test_the_menu_approves_new_plugins_then_asks_about_their_conflict(monkeypatch):
    monkeypatch.setattr(plugins, "reports", apply(component(), STACKABLE, approvals=plugins.Approvals()))
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
    plugins.save_approvals(allowed_both())
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=allowed_both()))
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
            "plugins_dir = Path(os.environ['SLIPAGENT_STATE_DIR']) / 'plugins'\n" + textwrap.dedent(body).strip(), "        "))
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
    assert len(applied) == 1 and "Wrote Widget.size" in menu.out.getvalue()

    files = {**STACKABLE, "c.py": PICK}
    approvals = allowed_both()
    approvals.decide("c.py", "Widget.size", False)
    monkeypatch.setattr(plugins, "reports", apply(component(), files, approvals=approvals))
    menu, _, _ = menu_for([], monkeypatch)
    menu.listing()
    assert menu.out.getvalue().splitlines()[1:] == [
        "  a.py → widgets", "    Widget.render: conflict unresolved, unstamped",
        "  b.py → widgets", "    Widget.render: conflict unresolved, unstamped",
        "  c.py → widgets", "    Widget.render: awaiting approval, unstamped", "    Widget.size: rejected, unstamped",
    ]
