"""交互入口、终端输入输出和异步事件唤醒。"""

import sys
from pathlib import Path

# Give direct execution the same module identity as package execution, so
# permission prompts share this module's ConsoleBroker and CLI_ACTIVE state.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "agent.Jarvis.v2"
if __name__ == "__main__":
    sys.modules[f"{__package__}.code"] = sys.modules[__name__]

import threading
import time

from .config import PROMPT, READLINE_PROMPT
from .context import block_type, update_context
from .runtime import agent_lock, agent_loop
from .scheduler import (
    cron_lock,
    cron_queue,
    has_pending_background,
    start_runtime_services,
)
from .teams import consume_lead_inbox, format_team_events
from .tools import trigger_hooks


try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

CLI_ACTIVE = False


class ConsoleBroker:
    """Serialize normal prompts and worker permission questions on one stdin."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reader = None
        self.display_prompt = PROMPT
        self.readline_prompt = READLINE_PROMPT

    def set_prompt(self, display_prompt: str, readline_prompt: str):
        self.display_prompt = display_prompt
        self.readline_prompt = readline_prompt

    def ask(self, prompt: str | None = None) -> str:
        with self._lock:
            active_prompt = self.readline_prompt if prompt is None else prompt
            return (self.reader or input)(active_prompt)


CONSOLE = ConsoleBroker()


def terminal_print(text: str):
    if threading.current_thread() is threading.main_thread() or not CLI_ACTIVE:
        print(text)
        return
    line = ""
    if READLINE_AVAILABLE:
        try:
            line = readline.get_line_buffer()
        except Exception:
            line = ""
    print(f"\r\033[K{text}")
    print(CONSOLE.display_prompt + line, end="", flush=True)


def print_turn_assistants(messages: list, turn_start: int):
    for msg in messages[turn_start:]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block_type(block) == "text":
                terminal_print(block["text"] if isinstance(block, dict) else block.text)


def async_event_loop(history: list, context: dict, session_state: dict):
    while True:
        time.sleep(1)
        with agent_lock:
            with cron_lock:
                fired = list(cron_queue)
            inbox = consume_lead_inbox(route_protocol=True)
            if not fired and not inbox and not has_pending_background():
                continue
            turn_start = len(history)
            scheduled_requests = []
            for job in fired:
                scheduled_requests.append(f"Run scheduled task: {job.prompt}")
                terminal_print(
                    f"  \033[35m[cron auto] {job.prompt[:60]}\033[0m")
            if inbox:
                history.append({"role": "user",
                                "content": format_team_events(inbox)})
                terminal_print(
                    f"  \033[33m[team auto] {len(inbox)} events\033[0m")
            active_request = (
                "\n".join(scheduled_requests)
                if scheduled_requests
                else session_state["active_user_request"]
            )
            agent_loop(history, context, active_request)
            context.update(update_context(context, history))
            print_turn_assistants(history, turn_start)


def main():
    global CLI_ACTIVE
    CLI_ACTIVE = True
    start_runtime_services()
    print("s15: integrated harness")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    session_state = {"active_user_request": "(no active user request)"}
    threading.Thread(target=async_event_loop,
                     args=(history, context, session_state), daemon=True).start()
    while True:
        try:
            query = CONSOLE.ask()
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip():
            continue
        if query.strip().lower() in ("q", "exit"):
            break
        with agent_lock:
            trigger_hooks("UserPromptSubmit", query)
            turn_start = len(history)
            session_state["active_user_request"] = query
            history.append({"role": "user", "content": query})
            agent_loop(history, context, query)
            context = update_context(context, history)
            print_turn_assistants(history, turn_start)
        print()


if __name__ == "__main__":
    main()
