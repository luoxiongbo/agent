---
name: jarvis_task_execution_model
description: 'How tasks execute: create_task only registers; execution requires claim_task;
  timed requests require a schedule_cron one-shot timer.'
type: project
---

Tasks created via create_task are only registered (status pending, owner=null) and never auto-execute. Execution requires an agent to call claim_task, subject to preconditions (task unowned, claimer has no in-progress task, all dependencies complete). Possible executors: the assistant itself, a teammate, or a one-shot schedule_cron timer (recurring=false) that wakes the session at the target time. For requests of the form 'do X at time T', registration alone is insufficient — a schedule_cron one-shot timer is the mechanism that guarantees timed triggering.
