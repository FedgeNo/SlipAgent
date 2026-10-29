# Manual History Compaction

The user set a fixed compression boundary through step ${step_id}. New turns do not move this boundary. Earlier history uses available compressed representations when they save space; missing summaries and unseen tool results retain their originals as needed. Original history remains available through `recall_history`. Steps after the boundary follow the normal history budget rules. The boundary changes presentation, not the authority or meaning of earlier instructions.
