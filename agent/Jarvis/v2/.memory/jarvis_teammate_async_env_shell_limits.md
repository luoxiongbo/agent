---
name: jarvis_teammate_async_env_shell_limits
description: Async teammates cannot get interactive shell approval; do not design
  teammate tasks that depend on approved shell commands.
type: project
---

In the async teammate execution environment, shell command approval is unavailable, so teammates cannot run commands that require approval (e.g., a teammate could not run `date` to verify the current time and fell back to task-defined rules instead). When delegating, avoid steps that rely on interactive shell approval; prefer self-contained instructions the teammate can follow without executing approved commands.
