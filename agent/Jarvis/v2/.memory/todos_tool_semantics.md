---
name: todos_tool_semantics
description: 'How the todos list works: session-level entries with text + status,
  separate from the Jarvis task system; update only at real progress nodes.'
type: project
---

Todos are a session-level work list storing only entry text + status (pending / in_progress / completed). They do not persist task details, execution results, or outputs, and are separate from the create_task system (.tasks/ persists owner, state machine, and dependencies to disk). Update todos only at real progress nodes: write the initial plan when receiving a multi-step task, mark a step in_progress when starting it, mark completed when a step finishes, and add/remove entries when the plan changes; do not re-update when there is no progress.
