<script lang="ts">
    import {request} from './api';
    let {query, activate}: {query: string; activate: (index: number) => void} = $props();
    type Match = {index: number; before: string; match: string; after: string};
    let items = $state<Match[]>([]);
    let cursor = $state<string | null>('0:0');
    let busy = $state(false);
    let error = $state('');
    let generation = 0;
    let controller: AbortController;

    async function search(next: string, current: number) {
        busy = true;
        const found: Match[] = [];
        try {
            let position: string | null = next;
            while (position && found.length < 100) {
                const result: {items: Match[]; nextCursor: string | null} = await request(
                    `search?query=${encodeURIComponent(query)}&cursor=${position}&limit=${100 - found.length}`, undefined, controller.signal);
                if (current !== generation) return;
                found.push(...result.items);
                position = result.nextCursor;
                items = [...found];
            }
            cursor = position;
            error = '';
        } catch (failure) { if (current === generation && !controller.signal.aborted) error = String(failure); }
        finally { if (current === generation) busy = false; }
    }
    $effect(() => {
        const value = query.trim();
        generation++;
        const current = generation;
        controller?.abort();
        controller = new AbortController();
        items = []; cursor = '0:0'; busy = false;
        const timer = setTimeout(() => { if (value) void search('0:0', current); }, 250);
        return () => { clearTimeout(timer); controller.abort(); };
    });
</script>

<div class="SearchResults">
    <div class="SearchHeading"><h2>Search Original History</h2><small>{busy ? 'Searching…' : `${items.length} results${cursor ? ' on this page' : ''}`}</small></div>
    {#if error}<p class="error">{error}</p>{/if}
    {#each items as match}
        <button class="SearchResult" onclick={() => activate(match.index)} title={`Go to message ${match.index}`}>
            <span class="MatchBefore"><span>{match.before}</span></span><mark>{match.match}</mark><span class="MatchAfter">{match.after}</span>
        </button>
    {/each}
    {#if !query.trim()}<p class="empty">Search every original prompt, response and tool result.</p>{:else if !busy && !items.length}<p class="empty">No matches found.</p>{/if}
    {#if cursor && !busy && items.length}<button class="LoadMore" onclick={() => search(cursor!, generation)}>Next Results →</button>{/if}
</div>
