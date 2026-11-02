<script lang="ts">
    import {createVirtualizer} from '@tanstack/svelte-virtual';
    import {untrack} from 'svelte';
    import type {BrowserNode} from './api';
    let {items, selected = '', choose, promote, title, scroll = 0, onscroll = (_: number) => {}}: {
        items: BrowserNode[]; selected?: string; choose: (node: BrowserNode) => void;
        promote: (node: BrowserNode) => void; title: string; scroll?: number; onscroll?: (value: number) => void;
    } = $props();
    let element = $state<HTMLDivElement>();
    const virtualizer = createVirtualizer<HTMLDivElement, HTMLButtonElement>({
        count: 0, getScrollElement: () => element ?? null, estimateSize: () => 76, overscan: 5,
    });
    $effect(() => {
        const count = items.length, scroller = element;
        untrack(() => $virtualizer.setOptions({count, getScrollElement: () => scroller ?? null}));
    });
    $effect(() => { if (element) element.scrollTop = scroll; });
    function focus(node: BrowserNode) { if (node.collection) promote(node); else choose(node); }
    function key(event: KeyboardEvent, index: number) {
        let next = index;
        if (event.key === 'ArrowDown') next++;
        else if (event.key === 'ArrowUp') next--;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = items.length - 1;
        else if (event.key === 'ArrowRight' || event.key === 'Enter' && event.altKey) {
            focus(items[index]); return;
        } else return;
        event.preventDefault();
        next = Math.max(0, Math.min(items.length - 1, next));
        choose(items[next]);
        $virtualizer.scrollToIndex(next);
        requestAnimationFrame(() => element?.querySelector<HTMLButtonElement>(`[data-index="${next}"]`)?.focus());
    }
</script>

<div class="ListScroll" bind:this={element} onscroll={() => { if (element) onscroll(element.scrollTop); }} aria-label={title}>
    <div class="VirtualList" style:height={`${$virtualizer.getTotalSize()}px`}>
        {#each $virtualizer.getVirtualItems().filter((row) => row.index < items.length) as row (items[row.index].id)}
            {@const node = items[row.index]}
            <button class="ObjectRow" class:selected={node.id === selected} data-index={row.index}
                style:transform={`translateY(${row.start}px)`}
                onclick={() => choose(node)} ondblclick={() => focus(node)} onkeydown={(event) => key(event, row.index)}
                aria-pressed={node.id === selected} title={node.label}>
                <span class="ObjectIcon">{node.kind === 'folder' ? '▱' : node.collection ? '◈' : '·'}</span>
                <span class="ObjectCopy"><strong>{node.label}</strong><small>{node.preview}</small></span>
                {#if node.collection}<span class="Chevron">›</span>{/if}
            </button>
        {/each}
    </div>
    {#if !items.length}<p class="empty">No items in this collection.</p>{/if}
</div>
