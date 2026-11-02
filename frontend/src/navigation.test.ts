import {describe, it, expect} from 'vitest';
import {Navigation} from './navigation';

describe('hierarchy navigation', () => {
    it('restores parent selection, page and scroll independently of Back', () => {
        const nav = new Navigation();
        nav.current = {root: '/turns', selected: '/turns/100', cursor: '100', scroll: 500};
        nav.enter('/turns/100');
        nav.enter('/turns/100/tool_results');
        expect(nav.up().root).toBe('/turns/100');
        expect(nav.up()).toEqual({root: '/turns', selected: '/turns/100', cursor: '100', scroll: 500});
        expect(nav.back().root).toBe('/turns/100');
    });

    it('can focus arbitrary depths and remembers separate collections', () => {
        const nav = new Navigation();
        nav.enter('/files/src');
        nav.current.scroll = 200;
        nav.enter('/tools');
        expect(nav.enter('/files/src').scroll).toBe(200);
        for (let depth = 0; depth < 40; depth++) nav.enter(nav.current.root + '/child');
        for (let depth = 0; depth < 40; depth++) nav.up();
        expect(nav.current.root).toBe('/files/src');
    });
});
