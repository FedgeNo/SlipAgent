export type SessionState = {
    sessionId: string; workspace: string; model: string; running: boolean;
    stopped: boolean; queuedInputs: number; changing: boolean; closed: boolean;
};
export type BrowserNode = {id: string; label: string; kind: string; collection: boolean; preview: string; actionValue?: string; enabled?: boolean};
export type Children = {node: string; parent: string | null; items: BrowserNode[]; nextCursor: string | null};
export type Detail = {text: string; more: boolean};
export type FeedItem = Detail & {index: number; id: string; role: string; toolCalls: {index: number; name: string}[]; toolCount: number};
export type FeedPage = {items: FeedItem[]; offset: number; total: number; sessionId: string};
export type Notification = {type: string; epoch: string; sequence: number; state?: SessionState;
    event?: {kind: string; text: string; step: number}};

const capability = location.hash.slice(1) || sessionStorage.getItem('slipagentToken') || '';
history.replaceState(null, '', location.pathname);
if (capability) sessionStorage.setItem('slipagentToken', capability);
export const clientId = sessionStorage.getItem('slipagentClient') || crypto.randomUUID();
sessionStorage.setItem('slipagentClient', clientId);

export async function request<T>(path: string, body?: object, signal?: AbortSignal): Promise<T> {
    const response = await fetch('/api/' + path, {
        method: body ? 'POST' : 'GET',
        headers: {Authorization: 'Bearer ' + capability, ...(body ? {'Content-Type': 'application/json'} : {})},
        body: body ? JSON.stringify(body) : undefined, signal,
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({detail: 'Access refused or session unavailable.'}));
        throw new Error(typeof error.detail === 'string' ? error.detail : 'Invalid request.');
    }
    return response.json();
}

export function observe(receive: (record: Notification) => void, status: (text: string) => void): () => void {
    let stopped = false;
    let socket: WebSocket;
    let timer: ReturnType<typeof setTimeout>;
    function connect() {
        if (stopped) return;
        if (!capability) { status('Open the complete /gui link to connect'); return; }
        status('Connecting');
        socket = new WebSocket(`ws://${location.host}/api/events`);
        socket.onopen = () => socket.send(JSON.stringify({type: 'authenticate', token: capability, clientId: clientId}));
        socket.onmessage = ({data}) => {
            const record: Notification = JSON.parse(data);
            status('Connected');
            receive(record);
        };
        socket.onclose = ({code}) => {
            if (stopped) return;
            if (code === 1008) { status('Access refused · reopen the current /gui link'); return; }
            status('Reconnecting');
            timer = setTimeout(connect, 2000);
        };
    }
    connect();
    return () => { stopped = true; clearTimeout(timer); socket?.close(); };
}
