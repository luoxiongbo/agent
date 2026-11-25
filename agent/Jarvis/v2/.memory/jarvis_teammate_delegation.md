---
name: jarvis_teammate_delegation
description: 'Delegating a task to a teammate: confirm with user, create, bind (auto-claims),
  execute, complete, then explicitly shut down.'
type: project
---

Workflow: confirm with the user first, then create a teammate with a role (e.g., executor) and an optional worktree → bind the task to it → teammate executes → complete_task. Binding auto-claims the task: it enters in_progress with the teammate as owner, so the teammate must not call claim_task again (the claim step collapses in the delegation path). A worktree is only needed to isolate file edits that could conflict; pure-output tasks run without one. After complete_task the teammate does not terminate — it emits an idle_notification ('Waiting for more work'); the coordinating agent must explicitly send a shutdown/recycle request once the task is done.
