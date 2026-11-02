<script lang="ts">
    import {onMount} from 'svelte';
    import {clientId, request} from './api';
    let {node, sessionId, done}: {node: string; sessionId: string; done: () => void} = $props();
    type Page = {state: string; path?: string; text?: string; pages?: number; page?: number;
        readOnly?: boolean; message?: string; error?: string};
    let editor = $state<Page>({state: 'waiting'});
    let text = $state('');
    let page = $state(0);
    let dirty = $state(false);
    let busy = $state(false);
    let error = $state('');
    let discard = $state(false);
    let closed = false;
    let leased = false;
    let timer: ReturnType<typeof setTimeout>;
    function modal(dialog: HTMLDialogElement) { dialog.showModal(); }
    function command(action: string, values: object = {}) {
        return request('editor', {action, clientId: clientId, sessionId: sessionId, ...values});
    }
    async function load(next = 0) {
        try {
            const result = await request<Page>(`editor?clientId=${clientId}&page=${next}`);
            if (closed) return;
            editor = result;
            if (result.state === 'open') { text = result.text || ''; page = next; }
            else if (result.state === 'waiting') timer = setTimeout(() => load(next), 400);
            else error = result.error || 'The editor is no longer available. Close it and reopen the file.';
        } catch (failure) { error = String(failure); }
    }
    async function move(next: number) {
        busy = true;
        try { await command('stage', {page, text}); await load(next); error = ''; }
        catch (failure) { error = String(failure); }
        finally { busy = false; }
    }
    async function save() {
        busy = true;
        try {
            await command('save', {page, text});
            dirty = false; error = '';
            editor.message = 'File saved. Close the editor to let the agent continue.';
        } catch (failure) { error = String(failure); }
        finally { busy = false; }
    }
    async function close(force = false) {
        if (busy) return;
        if (dirty && !force) { discard = true; return; }
        busy = true;
        clearTimeout(timer);
        try {
            if (leased) await command('close');
            closed = true; done();
        } catch (failure) { error = String(failure); }
        finally { busy = false; }
    }
    onMount(() => {
        void command('open', {node}).then(() => {
            leased = true;
            if (closed) void command('close').catch(() => {});
            else void load();
        }).catch((failure) => error = String(failure));
        return () => { clearTimeout(timer); if (!closed && leased) void command('close').catch(() => {}); closed = true; };
    });
</script>

<dialog class="EditorDialog" use:modal aria-label="File Editor" oncancel={(event) => { event.preventDefault(); void close(); }}
    onkeydown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 's') { event.preventDefault(); if (!busy && editor.state === 'open') void save(); } }}>
    <header><div><small>WORKSPACE EDITOR</small><h2>{editor.path?.split(/[\\/]/).at(-1) || 'Opening File'}</h2><p>{editor.path || decodeURIComponent(node)}</p></div><button aria-label="Close Editor" disabled={busy} onclick={() => close()}>×</button></header>
    {#if error}<p class="EditorError" role="alert">{error}</p>{/if}
    {#if discard}<div class="DiscardBar"><span>Discard unsaved edits?</span><button onclick={() => discard = false}>Keep Editing</button><button class="DangerButton" onclick={() => close(true)}>Discard &amp; Close</button></div>{/if}
    {#if editor.state === 'open'}
        <textarea class="FileInput" aria-label="File Contents" bind:value={text} readonly={editor.readOnly || busy} spellcheck="false" oninput={() => dirty = true}></textarea>
        <footer><div><button disabled={busy || !page} onclick={() => move(page - 1)}>← Previous Page</button><small>Page {page + 1} / {editor.pages}{dirty ? ' · unsaved edits' : ''}</small><button disabled={busy || page + 1 >= (editor.pages || 1)} onclick={() => move(page + 1)}>Next Page →</button></div><button class="Primary" disabled={busy || editor.readOnly} onclick={save}>{busy ? 'Saving…' : 'Save File'}</button></footer>
    {:else}<div class="EditorWait"><span>◷</span><h3>{error ? 'Unable to Open File' : 'Waiting for a Safe Editing Boundary'}</h3><p>{editor.message || 'The editor will open when the current turn finishes. Your agent is still running.'}</p></div>{/if}
    <div class="EditorNote">{editor.readOnly ? 'Read-only file · unsupported control characters' : editor.message || 'The agent waits while the editor is open. Saved edits are recorded in its history.'}</div>
</dialog>
