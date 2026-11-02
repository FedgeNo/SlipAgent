"""Offline browser-test server using the production CLI driver and stub provider."""

import asyncio
import io
import os
import signal
import tempfile
from pathlib import Path

from test_cli_e2e import StubOpenRouter, cli_environment, text_step
from slipagent import cli
from slipagent.gui import start_gui
from slipagent.types import Message, ToolCall


async def main():
    with tempfile.TemporaryDirectory(prefix='slipagent-browser-') as directory:
        root = Path(directory)
        os.environ['SLIPAGENT_STATE_DIR'] = str(root / 'state')
        environment = cli_environment()
        os.environ.clear()
        os.environ.update(environment)
        project = root / 'project'
        (project / 'src' / 'nested').mkdir(parents=True)
        (project / 'src' / 'nested' / 'hello.py').write_text('print("hello browser")\n')
        (project / 'README.md').write_text('# A disposable browser fixture\n')
        package = project / '.slipagent/skills/browser-fixture'
        package.mkdir(parents=True)
        (package / 'SKILL.md').write_text('---\nname: browser-fixture\ndescription: Disposable browser skill\n---\nUse the shared application owners.\n')
        with StubOpenRouter([text_step('The shared engine received your browser prompt.')] * 100) as stub:
            args = cli.build_parser().parse_args(['--no-mcp', '--no-reload', '--model', 'stub/one',
                                                  '--base-url', stub.base_url, '-w', str(project)])
            session = await cli.Launcher(args).build_session()
            session.renderer.stream = io.StringIO()
            for index in range(80):
                session.agent.messages.extend([Message.user(f'Original question {index}'),
                                                Message.assistant(f'Original answer {index}\n' + 'History remains available. ' * 30)])
            session.agent.history.sync(session.agent.messages)
            plan = {'objective': 'Build a visual workspace', 'stages': [
                {'task': 'Share the engine', 'status': 'complete', 'evidence': 'One application owner'},
                {'task': 'Connect the browser', 'status': 'in_progress', 'evidence': 'Offline browser checks'}]}
            session.agent.messages.extend([Message.assistant('Keep the existing plan', [ToolCall('fixture-plan', 'update_plan', plan)]),
                                            Message(role='tool', name='update_plan', tool_call_id='fixture-plan',
                                                    content={'tool': 'update_plan', 'call_id': 'fixture-plan', 'status': 'success', 'content': plan})])
            session.agent.history.sync(session.agent.messages)
            session.agent._persist()
            session.registry.resources.artifacts.save('Example retained output', 'Complete retained output for browser inspection.\n', kind='output')
            target = project / 'restore.txt'
            target.write_text('Before checkpoint\n')
            checkpoints = session.agent._checkpoints()
            checkpoints.begin(len(session.agent.history.steps) + 1)
            entry = checkpoints.prepare(target, b'After checkpoint\n')
            target.write_text('After checkpoint\n')
            checkpoints.complete(entry)
            checkpoints.finish()
            host = await start_gui(session, port=18763)
            host.token = 'browser-test-capability'

            async def read(*args):
                await asyncio.Event().wait()
            cli.Repl.read_line = read
            task = asyncio.create_task(cli.Repl(session).run())
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
            try:
                await task
            except asyncio.CancelledError:
                pass


if __name__ == '__main__':
    asyncio.run(main())
