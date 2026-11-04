# Explicit Conversation Compaction Notice

The user explicitly requested conversation compaction through step ${step_id}. This fixed boundary identifies that action; new turns do not move it. This notice remains present afterward and does not announce a new compaction on each request. If applicable instructions require an action after compaction, check history for whether it was already performed after this boundary instead of repeating it every turn.

Earlier history uses available compressed representations when they save space; missing summaries and unseen tool results retain their originals as needed. Original history remains available through `recall_history`. Steps after the boundary follow the normal rolling history budget rules. The boundary changes presentation, not the authority or meaning of earlier instructions. In the absence of this explicit notice, ordinary rolling compression does not announce a whole-conversation compaction or context reset.
