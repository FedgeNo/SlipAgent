"""`--update` installs a newer GitHub release through its own installer."""

import io
import asyncio
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import httpx

from slipagent import __version__, updater

COMMIT = 'a' * 40


def archive_with(files):
    def fetch(url, destination):
        assert url == updater.REPOSITORY + f'/archive/{COMMIT}.tar.gz'
        with tarfile.open(destination, "w:gz") as archive:
            for name, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return fetch


@pytest.fixture
def managed(tmp_path, monkeypatch):
    root = tmp_path / "install"
    monkeypatch.setattr(updater, "installation", lambda: (root, {"launcher": str(tmp_path / "bin" / "slipagent")}))
    return root


def test_newer_release_runs_its_installer_for_this_installation(managed, tmp_path):
    calls = []
    fetch = archive_with({"SlipAgent-master/pyproject.toml": 'version = "99.0.0"\n',
                          "SlipAgent-master/install.py": "print('installer')\n"})
    def run(command, cwd):
        calls.append(command)
        assert (Path(cwd) / "install.py").is_file()
        return subprocess.CompletedProcess(command, 0)
    assert updater.update(fetch=fetch, run=run, resolve=lambda: COMMIT) == 0
    [command] = calls
    assert command[1].endswith("install.py")
    assert command[2:] == ["--install-dir", str(managed), '--source-commit', COMMIT, "--bin-dir", str(tmp_path / "bin")]
    assert not list(updater.data_directory().glob("update-*"))


def test_current_release_does_not_reinstall(managed, monkeypatch):
    monkeypatch.setattr(updater, 'installation', lambda: (managed, {'commit': COMMIT}))
    assert updater.update(fetch=lambda *args: pytest.fail('downloaded an installed commit'),
                          run=lambda *args, **kwargs: pytest.fail("installer ran"), resolve=lambda: COMMIT) == 0


def test_unsafe_archive_is_rejected(managed, capsys):
    fetch = archive_with({"SlipAgent-master/../escape.py": "x"})
    assert updater.update(fetch=fetch, run=lambda *args, **kwargs: pytest.fail("installer ran"), resolve=lambda: COMMIT) == 1
    assert "unsafe archive entry" in capsys.readouterr().err


def test_development_checkout_is_not_updated(monkeypatch, capsys):
    monkeypatch.setattr(updater, "installation", lambda: None)
    assert updater.update(fetch=lambda *args: pytest.fail("downloaded")) == 2
    assert "not a managed installation" in capsys.readouterr().err


async def test_update_resolves_new_head_instead_of_startup_commit(managed, monkeypatch):
    monkeypatch.setattr(updater, 'installed_identity', lambda: {'commit': 'c' * 40, 'version': __version__})

    def handle(request):
        if request.url.path.endswith('/commits/master'):
            return httpx.Response(200, text='b' * 40)
        if request.url.path.endswith('/pyproject.toml'):
            return httpx.Response(200, text=f'version = "{__version__}"\n')
        return httpx.Response(200, json={'status': 'ahead'})

    observed = await updater.check_for_updates(transport=httpx.MockTransport(handle))
    calls = []
    fetch = archive_with({'SlipAgent-latest/pyproject.toml': f'version = "{__version__}"\n',
                          'SlipAgent-latest/install.py': 'pass\n'})

    def resolve():
        calls.append('resolve')
        return COMMIT

    def run(command, cwd):
        calls.append(command)
        assert command[command.index('--source-commit') + 1] == COMMIT != observed.commit
        return subprocess.CompletedProcess(command, 0)

    assert updater.update(fetch=fetch, resolve=resolve, run=run) == 0
    assert calls[0] == 'resolve'


@pytest.mark.parametrize('status,available', [('ahead', True), ('behind', False), ('identical', False), ('diverged', False)])
async def test_startup_check_uses_commit_ancestry_not_just_version(monkeypatch, status, available):
    monkeypatch.setattr(updater, 'installed_identity', lambda: {'commit': 'b' * 40, 'version': __version__})
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path.endswith('/commits/master'):
            return httpx.Response(200, text=COMMIT)
        if request.url.path.endswith('/pyproject.toml'):
            return httpx.Response(200, text=f'version = "{__version__}"\n')
        return httpx.Response(200, json={'status': status})

    found = await updater.check_for_updates(transport=httpx.MockTransport(handle))
    assert (found is not None) is available
    assert str(requests[1].url).endswith(f'/{COMMIT}/pyproject.toml')
    assert f"{'b' * 40}...{COMMIT}" in str(requests[2].url)


async def test_identical_commit_needs_only_one_request(monkeypatch):
    monkeypatch.setattr(updater, 'installed_identity', lambda: {'commit': COMMIT, 'version': __version__})
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, text=COMMIT)

    assert await updater.check_for_updates(transport=httpx.MockTransport(handle)) is None
    assert len(calls) == 1


async def test_update_check_times_out_and_closes_network_stream(monkeypatch):
    monkeypatch.setattr(updater, 'CHECK_TIMEOUT', .1)
    monkeypatch.setattr(updater, 'installed_identity', lambda: {'commit': COMMIT, 'version': __version__})
    closed = asyncio.Event()

    class Silent(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b''

        async def aclose(self):
            closed.set()

    with pytest.raises(TimeoutError):
        await updater.check_for_updates(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Silent())))
    assert closed.is_set()


def test_legacy_identity_migration_preserves_unrelated_metadata(tmp_path, monkeypatch):
    root = tmp_path / 'managed'
    root.mkdir()
    metadata = {'source': str(root / 'source'), 'launcher': 'custom', 'unrelated': 'keep'}
    monkeypatch.setattr(updater, 'installation', lambda: (root, dict(metadata)))
    found = updater.installed_identity()
    assert found['version'] == __version__ and found['commit'] is None
    assert found['unrelated'] == 'keep'
    assert json.loads((root / 'installation.json').read_text()) == found


async def test_update_notice_waits_until_startup_finishes(monkeypatch):
    from types import SimpleNamespace
    from slipagent import cli

    ready = asyncio.Event()
    output = []
    session = SimpleNamespace(extensions={}, renderer=SimpleNamespace(emit=lambda text, **kwargs: output.append(text)))

    async def check():
        return updater.Release(COMMIT, '99.0.0')

    monkeypatch.setattr(updater, 'check_for_updates', check)
    task = asyncio.create_task(cli.Launcher(None).startup_update_check(session, ready))
    await asyncio.sleep(0)
    assert not output and not task.done()
    ready.set()
    await task
    assert '/update' in output[0]


async def test_update_command_requests_clean_exit_before_running_installer(managed):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from slipagent.commands import CommandContext, UpdateCommand
    from slipagent.cli import Style

    session = SimpleNamespace(extensions={}, agent=SimpleNamespace(request_stop=Mock()),
                              renderer=SimpleNamespace(_queued_settings_pause=False))
    context = CommandContext(session, Style(False), io.StringIO())
    assert await UpdateCommand().run(context, '') is True
    assert session.extensions['update_requested']
    session.agent.request_stop.assert_called_once()


async def test_update_dispatch_runs_after_repl_closes_and_off_the_event_loop(monkeypatch):
    from types import SimpleNamespace
    import threading
    from slipagent import cli

    closed = False
    loop_thread = threading.get_ident()
    session = SimpleNamespace(extensions={'update_requested': True})

    async def build(*args, **kwargs):
        return session

    async def repl(self):
        nonlocal closed
        closed = True
        return 0

    def update():
        assert closed
        assert threading.get_ident() != loop_thread
        return 3

    monkeypatch.setattr(cli.Launcher, 'build_session', build)
    monkeypatch.setattr(cli.Repl, 'run', repl)
    monkeypatch.setattr(updater, 'update', update)
    assert await cli.Launcher(cli.build_parser().parse_args([])).dispatch(None) == 3
