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
          allow=None) -> list[plugins.PluginReport]:
    sources = {f"{plugins.PLUGIN_KEY}/{name}": textwrap.dedent(text).encode() for name, text in files.items()}
    return plugins.overlay({"widgets": module}.get, lambda name: source.encode() if name == "widgets" else None,
                           sources, core, allow)


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


def test_allow_filters_individual_overrides():
    module = component()
    report = apply(module, {"pick.py": '''
        from slipagent.plugins import replaces

        @replaces("widgets.Widget")
        class Widget:
            def render(self, text):
                return "plugin"

            def size(self):
                return 9
    '''}, allow=lambda plugin, member: member == "Widget.size")[0]
    assert module.Widget().render("x") == "<x>" and module.Widget().size() == 9
    assert [override.member for override in report.overrides] == ["size"]


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
        for name, text in (plugin_files or {}).items():
            (directory / name).write_text(textwrap.dedent(text))
        script = SETUP.replace("__BODY__", textwrap.indent(
            "plugins_dir = Path(os.environ['SLIPAGENT_STATE_DIR']) / 'plugins'\n" + textwrap.dedent(body).strip(), "        "))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(package_root)])}
        result = subprocess.run([sys.executable, "-c", script], cwd=project, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr

    return run


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
