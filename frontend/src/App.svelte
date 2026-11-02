<script lang="ts">
    import {onMount} from 'svelte';
    import {observe, request, type BrowserNode, type Children, type Detail, type HistoryReference, type SessionState} from './api';
    import {Navigation, type Position} from './navigation';
    import ListPane from './ListPane.svelte';
    import Feed from './Feed.svelte';
    import Search from './Search.svelte';
    import Editor from './Editor.svelte';

    const navigation = new Navigation();
    const sections = [
        {id: '/turns', icon: '◈', label: 'History'}, {id: '/sessions', icon: '▣', label: 'Sessions'},
        {id: '/files', icon: '▱', label: 'Files'}, {id: '/plan', icon: '☷', label: 'Plan'},
        {id: '/tools', icon: '⌘', label: 'Tools'},
        {id: '/models', icon: '◇', label: 'Models'}, {id: '/skills', icon: '◌', label: 'Skills'},
        {id: '/artifacts', icon: '≋', label: 'Outputs'},
        {id: '/checkpoints', icon: '↶', label: 'Restore'},
        {id: '/settings', icon: '⚙', label: 'Settings'},
    ];
    let sessionState = $state<SessionState>();
    let connection = $state('Connecting');
    let position = $state<Position>({...navigation.current});
    let canBack = $state(false);
    let left = $state<Children>();
    let right = $state<Children>();
    let selection = $state<BrowserNode>();
    let detail = $state<Detail>();
    let detailOffset = $state(0);
    let error = $state('');
    let runError = $state('');
    let notice = $state('');
    let prompt = $state('');
    let promptInput = $state<HTMLTextAreaElement>();
    let promptReference = $state<HistoryReference>();
    let sending = $state(false);
    let actionBusy = $state(false);
    let revision = $state(0);
    let activity = $state('Ready');
    let streaming = $state('');
    let reasoning = $state('');
    let previewElement = $state<HTMLDivElement>();
    let previewFollow = $state(true);
    let controls = $state<{settings: Record<string, boolean>; queue: [string, string][]}>({settings: {}, queue: []});
    let confirmAction = $state<{title: string; action: string; identifier?: string; text?: string} | null>(null);
    let renameText = $state('');
    let renameOpen = $state(false);
    let renameTarget = $state('');
    let rootRequest: AbortController;
    let detailRequest: AbortController;
    let refreshTimer: ReturnType<typeof setTimeout> | undefined;
    let pageCursors = $state<string[]>([]);
    let searchMode = $state(false);
    let searchQuery = $state('');
    let searchInput = $state<HTMLInputElement>();
    let jump = $state(-1);
    let editing = $state('');
    let planData = $state<{objective?: string; stages?: {task: string; status: string; evidence: string}[]}>({});
    function openSearch() { searchMode = true; requestAnimationFrame(() => searchInput?.focus()); }
    function modal(node: HTMLDialogElement) { node.showModal(); }

    const historyReference = $derived(detail?.historyReference ?? right?.historyReference ??
        (!selection ? left?.historyReference : undefined));
    const historyAction = $derived(historyReference?.target.section === 'all' && !historyReference.target.fieldPath.length ?
        'Ask About This Step' : historyReference?.target.callId && historyReference.target.fieldPath.length === 1 ?
        (historyReference.target.section === 'tool_calls' ? 'Ask About This Call' : 'Ask About This Result') : 'Ask About This');
    function askHistory() {
        if (!historyReference || historyReference.sessionId !== sessionState?.sessionId) return;
        promptReference = historyReference;
        searchMode = false;
        requestAnimationFrame(() => promptInput?.focus());
    }

    $effect(() => {
        streaming; reasoning;
        if (previewElement && previewFollow) previewElement.scrollTop = previewElement.scrollHeight;
    });

    function savePosition() { navigation.current = {...position, previous: [...pageCursors]}; }
    async function navigate(root: string, back = false) {
        savePosition();
        position = {...(back ? navigation.back() : navigation.enter(root))};
        canBack = navigation.canGoBack;
        pageCursors = [...(position.previous ?? [])];
        await loadRoot();
    }
    async function up() {
        savePosition();
        position = {...navigation.up()};
        canBack = navigation.canGoBack;
        pageCursors = [...(position.previous ?? [])];
        await loadRoot();
    }
    async function loadRoot(preserveSelection = false) {
        rootRequest?.abort();
        rootRequest = new AbortController();
        const signal = rootRequest.signal;
        try {
            const page = await request<Children>(`children?node=${encodeURIComponent(position.root)}&cursor=${encodeURIComponent(position.cursor)}`, undefined, signal);
            const previousIndex = left?.node === page.node ? left.items.findIndex((item) => item.id === position.selected) : -1;
            const selected = page.items.find((item) => item.id === position.selected) ??
                (previousIndex >= 0 ? page.items[Math.min(previousIndex, page.items.length - 1)] : undefined);
            left = page;
            if (position.root === '/plan') planData = await request('plan', undefined, signal);
            if (preserveSelection && selected?.id === position.selected) return;
            detailRequest?.abort();
            if (selected) await choose(selected);
            else { selection = undefined; right = undefined; detail = undefined; position.selected = ''; }
            error = '';
        } catch (failure) { if (!signal.aborted) error = String(failure); }
    }
    async function choose(node: BrowserNode, remember = true) {
        detailRequest?.abort();
        detailRequest = new AbortController();
        const signal = detailRequest.signal;
        if (remember) position.selected = node.id;
        selection = node;
        detail = undefined; right = undefined; detailOffset = 0;
        try {
            if (node.collection || node.kind === 'session') {
                right = await request<Children>(`children?node=${encodeURIComponent(node.id)}`, undefined, signal);
            } else {
                detail = await request<Detail>(`detail?node=${encodeURIComponent(node.id)}`, undefined, signal);
            }
            error = '';
        } catch (failure) { if (!signal.aborted) error = String(failure); }
    }
    async function chooseRight(node: BrowserNode) {
        if (node.collection && !['model', 'tool', 'artifact', 'skill', 'checkpoint'].includes(node.kind)) await navigate(node.id);
        else await choose(node, false);
    }
    async function inspect(node: string) {
        await navigate(node.includes('/content') ? node.slice(0, node.lastIndexOf('/')) : node);
    }
    async function detailPage(offset: number) {
        if (!selection) return;
        try {
            detail = await request<Detail>(`detail?node=${encodeURIComponent(selection.id)}&offset=${offset}`);
            detailOffset = offset;
        } catch (failure) { error = String(failure); }
    }
    async function rootPage(next: boolean) {
        if (next && left?.nextCursor) { pageCursors.push(position.cursor); position.cursor = left.nextCursor; }
        else if (!next) position.cursor = pageCursors.pop() ?? '';
        position.scroll = 0;
        await loadRoot();
    }
    async function action(name: string, values: object = {}) {
        if (!sessionState) return;
        actionBusy = true;
        try {
            await request('actions', {sessionId: sessionState.sessionId, action: name, ...values});
            controls = await request('controls');
            revision++;
            await loadRoot();
            error = '';
        } catch (failure) { error = String(failure); }
        finally { actionBusy = false; }
    }
    async function submit() {
        if (sending || !prompt.trim()) return;
        sending = true;
        try {
            await request('actions', {sessionId: promptReference?.sessionId ?? sessionState?.sessionId, action: 'submit',
                text: prompt, ...(promptReference ? {historyReference: promptReference.target} : {})});
            prompt = '';
            promptReference = undefined;
            controls = await request('controls');
            error = '';
        } catch (failure) { error = String(failure); }
        finally { sending = false; }
    }
    function changed() {
        if (refreshTimer) return;
        refreshTimer = setTimeout(() => {
            refreshTimer = undefined; revision++;
            void request<typeof controls>('controls').then((value) => controls = value).catch(() => {});
            if (position.root.startsWith('/turns') || position.root === '/plan' || position.root === '/sessions') void loadRoot(true);
        }, 300);
    }
    onMount(() => {
        const stop = observe((record) => {
            if (record.state) {
                const previous = sessionState?.sessionId;
                const switched = sessionState?.sessionId !== record.state.sessionId;
                sessionState = record.state;
                if (promptReference && promptReference.sessionId !== record.state.sessionId) {
                    promptReference = undefined;
                    notice = 'History reference cleared because the session changed. Your draft is preserved.';
                }
                if (switched || record.type === 'snapshot') {
                    revision++;
                    const section = position.root.split('/')[1];
                    if (previous && switched && position.root.startsWith(`/sessions/${previous}/`)) void navigate('/turns');
                    else if (previous && switched && ['turns', 'messages', 'plan', 'queue', 'artifacts', 'checkpoints'].includes(section)) void navigate('/' + section);
                    else void loadRoot();
                }
                changed();
            }
            const event = record.event;
            if (!event) return;
            if (event.kind === 'assistant_delta') streaming = (streaming + event.text).slice(-16384);
            if (event.kind === 'reasoning_delta') reasoning = (reasoning + event.text).slice(-16384);
            if (event.kind === 'step_start') { activity = `Working · turn ${event.step}`; streaming = ''; reasoning = ''; previewFollow = true; runError = ''; }
            if (event.kind === 'tool_start') activity = 'Running tools';
            if (event.kind === 'notice' || event.kind === 'warning') notice = event.text;
            if (event.kind === 'retry') { activity = event.text; streaming = ''; reasoning = ''; notice = event.text; }
            if (event.kind === 'retry_wait') activity = event.text ? `Retrying in ${event.step}s · ${event.text}` : 'Working';
            if (event.kind === 'error') runError = event.text;
            if (['step_end', 'stream_end', 'error'].includes(event.kind)) { activity = event.kind === 'error' ? event.text : 'Ready'; streaming = ''; reasoning = ''; }
            if (['user_message', 'user_message_sent', 'tool_end', 'assistant_text', 'step_end', 'user_queue_changed', 'session_title'].includes(event.kind)) changed();
        }, (text) => connection = text);
        return () => { stop(); clearTimeout(refreshTimer); rootRequest?.abort(); detailRequest?.abort(); };
    });
    const crumbs = $derived(position.root.split('/').filter(Boolean).map((label, index, parts) => ({
        label: decodeURIComponent(label), id: '/' + parts.slice(0, index + 1).join('/'),
    })));
</script>

<svelte:head><title>{sessionState?.running ? 'Working · ' : ''}SlipAgent</title></svelte:head>
<svelte:window onkeydown={(event) => { if (event.key === 'F3') { event.preventDefault(); openSearch(); } else if (event.key === 'Escape' && searchMode) { event.preventDefault(); searchMode = false; } }}/>
<div class="AppShell">
    <nav class="NavigationRail" aria-label="Workspace Sections">
        <div class="BrandMark" title="SlipAgent">✳</div>
        {#each sections as section}
            <button class:active={position.root.startsWith(section.id)} title={section.label}
                aria-label={section.label} onclick={() => navigate(section.id)}><span>{section.icon}</span><small>{section.label}</small></button>
        {/each}
        <div class="RailBottom"><span class="ConnectionDot" class:online={connection === 'Connected'} title={connection}></span></div>
    </nav>
    <main>
        <header class="AppHeader">
            <div><h1>{sessionState?.workspace.split(/[\\/]/).filter(Boolean).at(-1) || 'SlipAgent'}<span class="WorkspaceTag">WORKSPACE</span></h1><p title={sessionState?.workspace}>{sessionState?.workspace || connection}</p></div>
            <div class="ModelStatus"><span class="StatusPill" class:working={sessionState?.running}>{sessionState?.running ? '● Working' : '○ Ready'}</span><span>{sessionState?.model || 'Connecting…'}</span></div>
        </header>
        {#if error || runError}<div class="ErrorBanner" role="alert">{error || runError}<button aria-label="Dismiss Error" onclick={() => { error = ''; runError = ''; }}>×</button></div>{/if}
        {#if notice}<div class="NoticeBanner" role="status">{notice}<button aria-label="Dismiss Notice" onclick={() => notice = ''}>×</button></div>{/if}
        <div class="WorkspaceGrid">
            <section class="ChatColumn">
                <div class="ColumnHeading"><h2>Conversation</h2><div><button class="TextButton" title="Press F3 to search" onclick={openSearch}>Search</button><button class="TextButton" onclick={() => { renameOpen = true; renameText = ''; renameTarget = ''; }}>Rename</button><button class="TextButton" disabled={actionBusy || sessionState?.running} onclick={() => action('fork')}>Fork</button></div></div>
                <div class="FeedContainer">
                    <div class="FeedLayer" class:search-hidden={searchMode}>{#if sessionState}<Feed {revision} sessionId={sessionState.sessionId} {inspect} {jump}/>{/if}</div>
                    {#if searchMode}<Search query={searchQuery} activate={(index) => { jump = -1; searchMode = false; requestAnimationFrame(() => jump = index); }}/>{/if}
                </div>
                {#if streaming || reasoning}<div class="StreamPreview" bind:this={previewElement} onscroll={() => {
                    if (previewElement) previewFollow = previewElement.scrollHeight - previewElement.scrollTop - previewElement.clientHeight < 24;
                }}>{#if reasoning}<small>Thinking</small><pre>{reasoning}</pre>{/if}{#if streaming}<small>Streaming</small><pre>{streaming}</pre>{/if}</div>{/if}
                {#if controls.queue.length}<div class="InputQueuePanel"><small>QUEUED INPUT</small><button class="TextButton" onclick={() => navigate('/queue')}>View Full Queue</button>{#each controls.queue as [id, text]}<div><span>{text}</span><button aria-label="Discard Queued Prompt" onclick={() => action('discard', {identifier: id})}>×</button></div>{/each}</div>{/if}
                <form class="PromptForm" onsubmit={(event) => { event.preventDefault(); if (!searchMode) void submit(); }}>
                    {#if searchMode}
                        <div class="SearchInput"><input aria-label="Search History" bind:this={searchInput} bind:value={searchQuery} placeholder="Search all original history…" maxlength="256"/><button type="button" aria-label="Close Search" onclick={() => searchMode = false}>×</button></div>
                        <div class="ComposerFooter"><small>Search works while the agent runs · Esc returns to your prompt</small></div>
                    {:else}
                    {#if promptReference}<div class="HistoryContext"><span>About {promptReference.label}</span><button type="button" aria-label="Remove History Reference" onclick={() => promptReference = undefined}>×</button></div>{/if}
                    <textarea aria-label="Prompt" placeholder={promptReference ? 'Ask about this history selection…' : 'Describe what you want to build…'} bind:this={promptInput} bind:value={prompt} rows="3"
                        onkeydown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); void submit(); } }}></textarea>
                    <div class="ComposerFooter"><small>{sessionState?.running ? activity : connection === 'Connected' ? 'Enter to send · Shift+Enter for a new line' : connection}</small><div>
                        {#if sessionState?.running}<button type="button" class="TextButton" onclick={() => action('stop')}>Stop After Turn</button><button type="button" class="DangerButton" onclick={() => action('interrupt')}>Interrupt</button>{/if}
                        <button class="Primary" disabled={sending || !prompt.trim() || connection !== 'Connected'}>{sessionState?.running ? 'Queue ↗' : 'Send ↗'}</button>
                    </div></div>
                    {/if}
                </form>
            </section>
            <section class="HierarchyColumn">
                <div class="ColumnHeading HierarchyToolbar"><div><button aria-label="Back" disabled={!canBack} onclick={() => navigate('', true)}>←</button><button aria-label="Up One Level" disabled={position.root === '/'} onclick={up}>↑</button></div><h2>Explore</h2><button aria-label="Refresh Collection" onclick={() => loadRoot()}>↻</button></div>
                <nav class="BreadcrumbNavigation" aria-label="Hierarchy Path"><button onclick={() => navigate('/')}>⌂</button>{#each crumbs as crumb}<span>›</span><button onclick={() => navigate(crumb.id)}>{crumb.label}</button>{/each}</nav>
                {#if left}<ListPane title="Collection" items={left.items} selected={position.selected} choose={choose} promote={(node) => navigate(node.id)} scroll={position.scroll} onscroll={(value) => position.scroll = value}/>{:else}<p class="empty">Loading collection…</p>{/if}
                <div class="PageNavigation"><button disabled={!pageCursors.length && !position.cursor} onclick={() => rootPage(false)}>← Previous</button><small>{left?.items.length ?? 0} items</small><button disabled={!left?.nextCursor} onclick={() => rootPage(true)}>Next →</button></div>
                <div class="NavigationHint">Click to inspect · double-click to focus<br>↑ Up returns to the parent collection</div>
            </section>
            <section class="DetailColumn">
                <div class="ColumnHeading"><h2>{selection?.label || 'Details'}</h2>{#if selection?.collection}<button class="TextButton" onclick={() => navigate(selection!.id)}>Focus ↗</button>{/if}</div>
                {#if historyReference && historyReference.sessionId === sessionState?.sessionId}<div class="DetailActions"><button class="Primary" disabled={sending} onclick={askHistory}>{historyAction}</button><small>{historyReference.label}</small></div>{/if}
                {#if selection?.kind === 'file'}<div class="FileActions"><button class="TextButton" onclick={() => choose(selection!)}>View</button><button class="TextButton" onclick={() => editing = selection!.id}>Edit</button></div>{/if}
                {#if selection?.kind === 'model'}<div class="DetailActions"><button class="Primary" disabled={sessionState?.running || actionBusy} onclick={() => action('model', {identifier: selection!.actionValue})}>Select Model</button><small>{selection.preview}</small></div>{/if}
                {#if selection?.kind === 'optimal'}<div class="DetailActions"><button class="Primary" disabled={sessionState?.running || actionBusy} onclick={() => action('optimal')}>Enable Optimal Selection</button></div>{/if}
                {#if selection?.kind === 'skill'}<div class="DetailActions"><button class="Primary" disabled={sessionState?.running || actionBusy || !selection.enabled} onclick={() => action('skill', {identifier: selection!.actionValue})}>Use Skill</button><button class="TextButton" disabled={sessionState?.running || actionBusy} onclick={() => action('skill')}>Clear Skill</button></div>{/if}
                {#if selection?.kind === 'checkpoint'}<div class="DetailActions"><button class="DangerButton" disabled={sessionState?.running || actionBusy} onclick={() => confirmAction = {title: 'Restore These File Edits?', action: 'rewind', identifier: selection!.actionValue}}>Restore From Here</button><small>Restores this checkpoint and later file edits. Conversation history is preserved.</small></div>{/if}
                {#if position.root === '/plan' && !selection && planData.stages}
                    <div class="PlanBoard"><small>WORKING PLAN</small><h3>{planData.objective}</h3><progress max={planData.stages.length} value={planData.stages.filter((stage) => stage.status === 'complete').length}></progress>
                        {#each planData.stages as stage, index}<div class="PlanStage" class:complete={stage.status === 'complete'}><span>{stage.status === 'complete' ? '✓' : stage.status === 'in_progress' ? '◷' : '○'}</span><div><button class="TextButton" onclick={() => navigate(`/plan/stages/${index}`)}>{stage.task}</button><small>{stage.status.replaceAll('_', ' ')}</small>{#if stage.evidence}<details><summary>Evidence</summary><pre>{stage.evidence}</pre></details>{/if}</div></div>{/each}
                    </div>
                {/if}
                {#if position.root === '/settings' || selection?.id.startsWith('/settings')}
                    <div class="SettingList">{#each Object.entries(controls.settings) as [setting, enabled]}<label><span><strong>{setting}</strong><small>{setting === 'danger' ? 'Allow access beyond the workspace' : setting === 'planning' ? 'Keep a working plan across turns' : setting === 'hashline' ? 'Use hash-anchored file editing' : 'Include recent archived thoughts'}</small></span><input type="checkbox" checked={enabled} disabled={sessionState?.running || actionBusy} onchange={() => { if (setting === 'danger' && !enabled) confirmAction = {title: 'Enable Danger Mode?', action: 'setting', text: setting}; else void action('setting', {setting, enabled: !enabled}); }}/></label>{/each}</div>
                {:else if selection?.kind === 'session'}
                    <div class="DetailActions">
                        <p>{selection.preview}</p>
                        <button class="Primary" disabled={sessionState?.running || actionBusy} onclick={() => action('resume', {identifier: decodeURIComponent(selection!.id.split('/').at(-1)!)})}>Resume Session</button>
                        <button disabled={sessionState?.running || actionBusy} onclick={() => { renameTarget = decodeURIComponent(selection!.id.split('/').at(-1)!); renameText = ''; renameOpen = true; }}>Rename Session</button>
                        <button class="DangerButton" disabled={sessionState?.running || actionBusy} onclick={() => { const identifier = decodeURIComponent(selection!.id.split('/').at(-1)!); confirmAction = {title: 'Delete This Saved Session?', action: identifier === sessionState?.sessionId ? 'delete_current' : 'delete_saved', identifier}; }}>Delete Session</button>
                    </div>
                {/if}
                {#if right}
                    <ListPane title="Children" items={right.items} choose={chooseRight} promote={(node) => navigate(node.id)}/>
                    {#if right.nextCursor}<button class="LoadMore" onclick={() => { if (selection) void navigate(selection.id); }}>Focus Collection to Browse All →</button>{/if}
                {:else if detail}
                    <div class="DetailLabel">ORIGINAL CONTENT <span>{selection?.kind}</span></div>
                    <pre class="OriginalText">{detail.text}</pre>
                    <div class="PageNavigation"><button disabled={!detailOffset} onclick={() => detailPage(Math.max(0, detailOffset - 16384))}>← Previous</button><small>{detailOffset.toLocaleString()}–{(detailOffset + detail.text.length).toLocaleString()}</small><button disabled={!detail.more} onclick={() => detailPage(detailOffset + 16384)}>Next →</button></div>
                {:else if !right && !selection && !(position.root === '/plan' && planData.stages)}<div class="DetailEmpty"><span>◈</span><h3>Follow the Structure</h3><p>Select an item to see its children or content. Focus any collection to explore another level.</p></div>{/if}
                <div class="SessionActions"><button class="TextButton" disabled={sessionState?.running || actionBusy} onclick={() => action('new_task')}>New Task</button><button class="TextButton" disabled={sessionState?.running || actionBusy} onclick={() => confirmAction = {title: 'Reset This Conversation?', action: 'reset'}}>Reset Conversation</button></div>
            </section>
        </div>
    </main>
</div>
{#if editing && sessionState}<Editor node={editing} sessionId={sessionState.sessionId} done={() => { editing = ''; revision++; if (selection?.kind === 'file') void choose(selection); }}/>{/if}
{#if confirmAction}
    <dialog class="ConfirmationDialog" use:modal aria-label={confirmAction.title} onclose={() => confirmAction = null}><h2>{confirmAction.title}</h2><p>{confirmAction.action === 'reset' ? 'This clears the active conversation and starts a new session.' : confirmAction.action === 'setting' ? 'This lets tools access paths outside the workspace.' : confirmAction.action === 'rewind' ? 'Workspace files will be restored through this checkpoint. Changes outside the recorded edits are checked before restoration.' : 'The saved session and its retained records will be removed.'}</p><div class="ModalActions"><button onclick={() => confirmAction = null}>Cancel</button><button class="DangerButton" onclick={() => { const pending = confirmAction!; confirmAction = null; void action(pending.action, pending.action === 'setting' ? {setting: pending.text, enabled: true} : {identifier: pending.identifier}); }}>Confirm</button></div></dialog>
{/if}
{#if renameOpen}
    <dialog class="ConfirmationDialog" use:modal aria-label="Rename Session" onclose={() => renameOpen = false}><form onsubmit={(event) => { event.preventDefault(); renameOpen = false; void action(renameTarget ? 'rename_saved' : 'rename', {text: renameText, ...(renameTarget ? {identifier: renameTarget} : {})}); }}><h2>Rename Session</h2><input aria-label="Session Name" bind:value={renameText} placeholder="A name for this work"/><div class="ModalActions"><button type="button" onclick={() => renameOpen = false}>Cancel</button><button class="Primary" disabled={!renameText.trim()}>Rename</button></div></form></dialog>
{/if}
