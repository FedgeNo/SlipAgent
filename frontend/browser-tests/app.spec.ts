import {test, expect, type Page} from '@playwright/test';

async function live_events(page: Page) {
    let send: (kind: string, text: string) => void = () => { throw new Error('Socket not connected'); };
    await page.routeWebSocket('**/api/events', (socket) => {
        socket.connectToServer();
        send = (kind, text) => socket.send(JSON.stringify({type: 'event', event: {kind, text, step: 1}}));
    });
    await page.reload();
    await expect(page.locator('.ConnectionDot')).toHaveClass(/online/);
    return send;
}

test.beforeEach(async ({page}) => {
    await page.goto('/#browser-test-capability');
    await expect(page.getByRole('button', {name: 'Send ↗', exact: true})).toBeVisible();
});

test('browses original history and arbitrary file depths with Up and Back', async ({page}) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await expect(page.locator('.ToolResultRow').first()).toContainText('update_plan');
    await expect(page.locator('.MessageCard.tool pre')).toHaveCount(0);
    await expect(page.locator('.ObjectRow').first()).toBeVisible();
    await page.locator('.HierarchyColumn .ObjectRow').first().click();
    await expect(page.locator('.DetailColumn .ObjectRow').filter({hasText: 'response'})).toBeVisible();
    await page.locator('.DetailColumn .ObjectRow').filter({hasText: 'response'}).click();
    await expect(page.locator('.OriginalText')).toContainText('Original answer 0');
    await page.getByRole('button', {name: 'Files', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'src'}).dblclick();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'nested'}).dblclick();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'hello.py'}).click();
    await expect(page.locator('.OriginalText')).toContainText('hello browser');
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'hello.py'}).dblclick();
    await expect(page.locator('.OriginalText')).toContainText('hello browser');
    await page.getByRole('button', {name: 'Up One Level'}).click();
    await expect(page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'nested'})).toHaveAttribute('aria-pressed', 'true');
    await page.getByRole('button', {name: 'Back', exact: true}).click();
    await expect(page.locator('.OriginalText')).toContainText('hello browser');
    expect(errors).toEqual([]);
    await page.screenshot({path: 'test-results/gui-workspace.png', fullPage: true});
});

test('submits through the shared CLI engine and remains bounded after refresh', async ({page}) => {
    await page.getByRole('textbox', {name: 'Prompt', exact: true}).fill('A prompt sent from the GUI');
    await page.getByRole('button', {name: 'Send ↗', exact: true}).click();
    await expect(page.locator('.MessageCard').filter({hasText: 'The shared engine received your browser prompt.'})).toBeVisible({timeout: 20000});
    await expect(page.locator('.MessageCard').filter({hasText: 'The shared engine received your browser prompt.'}).locator('.ResponseSource')).toContainText('openrouter · stub/one');
    await page.reload();
    await expect(page.locator('.MessageCard').filter({hasText: 'The shared engine received your browser prompt.'})).toBeVisible();
    expect(await page.locator('.FeedPosition').count()).toBeLessThan(30);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await expect(page.getByRole('checkbox')).toHaveCount(4);
});

test('searches original history with F3 and jumps back into the feed', async ({page}) => {
    await page.keyboard.press('F3');
    await page.getByRole('textbox', {name: 'Search History'}).fill('Original question 3');
    await expect(page.locator('.SearchResult').first()).toBeVisible();
    await expect(page.locator('.SearchResult mark').first()).toHaveText('Original question 3');
    await page.locator('.SearchResult').first().click();
    await expect(page.locator('.MessageCard.user pre').filter({hasText: /^Original question 3$/})).toBeVisible();
});

test('edits a workspace file and views the saved original', async ({page}) => {
    await page.getByRole('button', {name: 'Files', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'src'}).dblclick();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'nested'}).dblclick();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'hello.py'}).click();
    await page.getByRole('button', {name: 'Edit', exact: true}).click();
    await expect(page.getByRole('textbox', {name: 'File Contents'})).toBeVisible();
    await page.getByRole('textbox', {name: 'File Contents'}).fill('print("edited in browser")\n');
    await page.getByRole('button', {name: 'Save File', exact: true}).click();
    await expect(page.locator('.EditorNote')).toContainText('File saved');
    await page.getByRole('button', {name: 'Close Editor'}).click();
    await expect(page.locator('.OriginalText')).toContainText('edited in browser');
});

test('uses the existing plan, skills, models and retained output hierarchies', async ({page}) => {
    await page.getByRole('button', {name: 'Plan', exact: true}).click();
    await expect(page.locator('.PlanBoard')).toContainText('Build a visual workspace');
    await expect(page.locator('.PlanStage.complete')).toContainText('Share the engine');
    await page.getByRole('button', {name: 'Skills', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'browser-fixture'}).click();
    await page.getByRole('button', {name: 'Use Skill', exact: true}).click();
    await page.getByRole('button', {name: 'Outputs', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'Example retained output'}).click();
    await page.locator('.DetailColumn .ObjectRow').filter({hasText: 'output'}).click();
    await expect(page.locator('.OriginalText')).toContainText('Complete retained output');
    await page.getByRole('button', {name: 'Models', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'Openrouter'}).click();
    await page.locator('.DetailColumn .ObjectRow').first().click();
    await expect(page.getByRole('button', {name: 'Select Model', exact: true})).toBeVisible();
    await page.getByRole('button', {name: 'Select Model', exact: true}).click();
    await expect(page.locator('.ErrorBanner')).toHaveCount(0);
});

test('restores checkpointed files after confirmation without resetting history', async ({page}) => {
    await page.getByRole('button', {name: 'Restore', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').first().click();
    await page.getByRole('button', {name: 'Restore From Here', exact: true}).click();
    await page.getByRole('dialog').getByRole('button', {name: 'Confirm', exact: true}).click();
    await page.getByRole('button', {name: 'Files', exact: true}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'restore.txt'}).click();
    await expect(page.locator('.OriginalText')).toContainText('Before checkpoint');
    await expect(page.locator('.MessageCard').filter({hasText: 'Original answer 79'})).toBeVisible();
});

test('starts a fresh conversation while retaining readable session instructions', async ({page}) => {
    await page.getByRole('button', {name: 'Reset Conversation', exact: true}).click();
    await page.getByRole('dialog').getByRole('button', {name: 'Confirm', exact: true}).click();
    await expect(page.getByRole('heading', {name: 'What Are We Building?'})).toBeVisible();
    await expect(page.getByRole('button', {name: 'Read Full Instructions →'})).toBeVisible();
    await page.getByRole('button', {name: 'Read Full Instructions →'}).click();
    await page.locator('.HierarchyColumn .ObjectRow').filter({hasText: 'content'}).click();
    await expect(page.locator('.OriginalText')).not.toBeEmpty();
});

test('keeps run errors and notices visible after the agent becomes idle', async ({page}) => {
    const send = await live_events(page);
    send('warning', 'The provider reported a warning.');
    await expect(page.getByRole('status')).toContainText('The provider reported a warning.');
    send('error', 'The provider request failed.');
    await expect(page.getByRole('alert')).toContainText('The provider request failed.');
    await page.getByRole('button', {name: 'Refresh Collection'}).click();
    await expect(page.getByRole('alert')).toContainText('The provider request failed.');
    await page.getByRole('button', {name: 'Dismiss Error'}).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', {name: 'Dismiss Notice'}).click();
    await expect(page.getByRole('status')).toHaveCount(0);
});

test('streams reasoning and follows long previews while preserving manual scroll', async ({page}) => {
    const send = await live_events(page);
    send('step_start', '');
    send('reasoning_delta', 'Inspecting the problem.');
    await expect(page.locator('.StreamPreview')).toContainText('Thinking');
    send('reasoning_delta', ' Checking the implementation.');
    await expect(page.locator('.StreamPreview pre').first()).toHaveText('Inspecting the problem. Checking the implementation.');
    send('assistant_delta', Array.from({length: 40}, (_,index) => `Streamed line ${index}`).join('\n'));
    const preview = page.locator('.StreamPreview');
    await expect(preview).toContainText('Streamed line 39');
    await expect.poll(() => preview.evaluate((element) => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThan(3);
    await preview.evaluate((element) => { element.scrollTop = 0; });
    await preview.dispatchEvent('scroll');
    await expect.poll(() => preview.evaluate((element) => element.scrollTop)).toBe(0);
    send('assistant_delta', '\nMore streamed output');
    await expect(preview).toContainText('More streamed output');
    await expect.poll(() => preview.evaluate((element) => element.scrollTop)).toBe(0);
    await preview.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await preview.dispatchEvent('scroll');
    send('assistant_delta', '\nLatest streamed output');
    await expect(preview).toContainText('Latest streamed output');
    await expect.poll(() => preview.evaluate((element) => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThan(3);
    send('stream_end', '');
    await expect(preview).toHaveCount(0);
    send('step_start', '');
    send('reasoning_delta', 'Fresh reasoning');
    await expect(page.locator('.StreamPreview pre')).toHaveText('Fresh reasoning');
    send('retry', 'Trying again');
    await expect(preview).toHaveCount(0);
});
