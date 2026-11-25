---
name: jarvis_task_lifecycle_design
description: Implementation details of the task lifecycle in agent/Jarvis/v2/tasks.py
  (create → claim → complete).
type: project
---

States: pending → in_progress → completed; IDs of form task_<hex>; persisted to .tasks/ via atomic writes (temp file + os.replace) under task_store_lock(). create_task(subject, description) creates a pending task with owner=None; update_task can add blockedBy dependencies with cycle detection via _task_depends_on. claim_task(task_id, owner) requires the task be pending/unowned, the owner has no in-progress task (_owner_in_progress), and all blockedBy dependencies are completed (can_start); it resolves a Git worktree cwd via task_worktree_cwd, sets in_progress, registers in teammate_assignments, and calls advance_assignment_version. complete_task(task_id, owner) requires in_progress status and caller == owner, checks teams.plan_gates, marks completed, and returns newly unblocked tasks; release_completed_assignment frees the owner's cwd lease and resets plan gates at model-turn boundaries.
