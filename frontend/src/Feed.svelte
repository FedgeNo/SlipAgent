<script lang="ts">
    import {createVirtualizer} from '@tanstack/svelte-virtual';
    import {tick, untrack} from 'svelte';
    import {request, type FeedPage, type FeedItem} from './api';
    let {revision, sessionId, inspect, jump = -1}: {
        revision: number; sessionId: string; inspect: (node: string) => void; jump?: number;
    } = $props();
    let element = $state<HTMLDivElement>();
    let total = $state(0);
    let cache = $state(new Map<number, FeedItem>());
    let follow = $state(true);
    let error = $state('');
    let generation = 0;
    const pages = new Map<number, number>();
    const pending = new Set<number>();
    const pageSize = 40;
    const virtualizer = createVirtualizer<HTMLDivElement, HTMLDivElement>({
        count: 0, getScrollElement: () => element ?? null, estimateSize: () => 150, overscan: 4,
        getItemKey: (index) => `${sessionId}:${index}`,
    });
    $effect(() => {
        const count = total, scroller = element;
        untrack(() => $virtualizer.setOptions({count, getScrollElement: () => scroller ?? null}));
    });

    async function load(offset: number, refresh = false) {
        if (pending.has(offset) || !refresh && pages.has(offset)) return;
        const current = generation;
        pending.add(offset);
        try {
            const page = await request<FeedPage>(`feed?offset=${offset}&limit=${pageSize}`);
            if (current !== generation || page.sessionId !== sessionId) return;
            total = page.total;
            const next = new Map(cache);
            for (const item of page.items) next.set(item.index, item);
            pages.set(offset, Date.now());
            if (pages.size > 6) {
                const oldest = [...pages].sort((a, b) => a[1] - b[1])[0][0];
                pages.delete(oldest);
                for (let i = oldest; i < oldest + pageSize; i++) next.delete(i);
            }
            cache = next;
            error = '';
            await tick();
            if (follow && total) $virtualizer.scrollToIndex(total - 1, {align: 'end'});
        } catch (failure) { error = String(failure); }
        finally { if (current === generation) pending.delete(offset); }
    }

    $effect(() => {
        sessionId;
        generation++;
        cache = new Map(); pages.clear(); pending.clear(); total = 0;
        follow = true;
        void refresh();
    });
    async function refresh() {
        const page = await request<FeedPage>('feed?limit=1').catch(() => null);
        if (!page || page.sessionId !== sessionId) return;
        total = page.total;
        const offset = Math.max(0, Math.floor((total - 1) / pageSize) * pageSize);
        await load(offset, true);
    }
    $effect(() => { revision; void refresh(); });
    $effect(() => {
        const rows = $virtualizer.getVirtualItems();
        for (const row of rows) void load(Math.floor(row.index / pageSize) * pageSize);
    });
    $effect(() => {
        if (jump >= 0) {
            follow = false;
            void load(Math.floor(jump / pageSize) * pageSize).then(() => $virtualizer.scrollToIndex(jump, {align: 'center'}));
        }
    });
    function measure(node: HTMLDivElement) { $virtualizer.measureElement(node); }
</script>

<div class="FeedScroll" bind:this={element} onscroll={() => { if (element) follow = element.scrollHeight - element.scrollTop - element.clientHeight < 100; }}>
    {#if error}<p class="error">{error}</p>{/if}
    {#if !total || (total === 1 && cache.get(0)?.role === 'system')}<div class="FeedEmpty"><span class="EmptyMark">✳</span><h2>What Are We Building?</h2><p>Send a prompt below. Your session and tools are shared with the terminal.</p></div>{/if}
    <div class="VirtualFeed" style:height={`${$virtualizer.getTotalSize()}px`}>
        {#each $virtualizer.getVirtualItems() as row (row.key)}
            <div class="FeedPosition" data-index={row.index} use:measure style:transform={`translateY(${row.start}px)`}>
                {#if cache.get(row.index)}
                    {@const item = cache.get(row.index)!}
                    <article class="MessageCard" class:user={item.role === 'user'} class:tool={item.role === 'tool'}>
                        <header class:ToolResultRow={item.role === 'tool'}><span class:failed={item.toolResult?.status === 'error'}>
                            {#if item.role === 'tool'}{item.toolResult?.status === 'success' ? '✓' : item.toolResult?.status === 'error' ? '✗' : '·'} {item.toolResult?.name || 'Tool Result'} · {item.toolResult?.status === 'success' ? 'Completed' : item.toolResult?.status === 'error' ? 'Failed' : 'Result'}
                            {:else}{item.role === 'assistant' ? 'SlipAgent' : item.role === 'system' ? 'Session Instructions' : item.role}{/if}
                        </span><button class="TextButton" onclick={() => inspect(item.id)}>Inspect ↗</button></header>
                        {#if item.role !== 'tool'}
                        {#if item.historyContext}<div class="HistoryContextLabel">About {item.historyContext}</div>{/if}
                        {#if item.source && (item.source.model || item.source.provider)}<div class="ResponseSource" title="Response Provider and Model">{[item.source.provider, item.source.model].filter(Boolean).join(' · ')}</div>{/if}
                        {#if item.role === 'system'}<button class="TextButton" onclick={() => inspect(item.id + '/content')}>Read Full Instructions →</button>{:else if item.text}<pre>{item.text}</pre>{/if}
                        {#if item.toolCalls.length}<div class="ToolCards">{#each item.toolCalls as call}<button onclick={() => inspect(`${item.id}/tool_calls/${call.index}`)}><span>⌘</span>{call.name}<span>↗</span></button>{/each}</div>{/if}
                        {#if item.toolCount > item.toolCalls.length}<button class="TextButton" onclick={() => inspect(item.id + '/tool_calls')}>Inspect All {item.toolCount} Calls →</button>{/if}
                        {#if item.more}<button class="TextButton" onclick={() => inspect(item.id + '/content')}>Read Full Content →</button>{/if}
                        {/if}
                    </article>
                {:else}<div class="MessageCard loading">Loading original record…</div>{/if}
            </div>
        {/each}
    </div>
</div>
{#if !follow && total}<button class="LatestButton" onclick={() => { follow = true; $virtualizer.scrollToIndex(total - 1, {align: 'end'}); }}>Latest ↓</button>{/if}
