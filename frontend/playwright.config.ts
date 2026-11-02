import {defineConfig} from '@playwright/test';
import {resolve} from 'node:path';

export default defineConfig({
    testDir: './browser-tests', workers: 1, fullyParallel: false,
    use: {baseURL: 'http://127.0.0.1:18763', viewport: {width: 1500, height: 950}},
    webServer: {
        command: '../.venv/bin/python ../tests/gui-fixture.py',
        url: 'http://127.0.0.1:18763', timeout: 60000, reuseExistingServer: false,
        env: {PYTHONPATH: [resolve('../tests/offline'), resolve('../tests'), resolve('../src')].join(':'),
              SLIPAGENT_NO_DOTENV: '1', SLIPAGENT_NO_UPDATE_CHECK: '1'},
    },
});
