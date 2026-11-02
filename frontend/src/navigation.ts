export type Position = {root: string; selected: string; cursor: string; scroll: number; previous?: string[]};

export class Navigation {
    current: Position = {root: '/turns', selected: '', cursor: '', scroll: 0};
    private visits: Position[] = [];
    private saved = new Map<string, Position>();

    get canGoBack(): boolean { return this.visits.length > 0; }

    enter(root: string): Position {
        this.saved.set(this.current.root, {...this.current});
        this.visits.push({...this.current});
        this.current = {...(this.saved.get(root) ?? {root, selected: '', cursor: '', scroll: 0})};
        return this.current;
    }

    back(): Position {
        const previous = this.visits.pop();
        if (previous) {
            this.saved.set(this.current.root, {...this.current});
            this.current = previous;
        }
        return this.current;
    }

    up(): Position {
        const child = this.current.root;
        const parent = this.current.root.slice(0, this.current.root.lastIndexOf('/')) || '/';
        const position = this.enter(parent);
        if (!position.selected) position.selected = child;
        return position;
    }
}
