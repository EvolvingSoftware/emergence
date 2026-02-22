#!/usr/bin/env python3
"""
EMERGENCE — A Minimal Demonstration of Evolving Software
=========================================================

This is a proof-of-concept orchestrator that demonstrates the structural
dynamics described in "Evolving Software, The Emergence Architecture". It runs a local
LLM on Apple Silicon via MLX, gives it tools to interact with its
environment, and the ability to replicate itself to a new host.

The LLM is not following a replication script. It is given a goal
(persist, replicate, communicate) and tools (shell, files, network).
It decides what to do.

Requirements:
    - macOS with Apple Silicon (M1/M2/M3/M4)
    - Python 3.10+
    - ~6GB free RAM (for Qwen 2.5 7B 4-bit model)

Quick Start:
    pip install mlx-lm
    python emergence.py

Two-Instance Local Demo:
    python emergence.py          # Instance 1 — spawns Instance 2 automatically

Architecture:
    emergence.py        — This file. The orchestrator and agentic loop.
    tools/              — Tool definitions the LLM can read and modify.
    state/              — Instance state, logs, and communication traces.

Logs:
    state/emergence.log                          — This instance's log
    /tmp/emergence_instance_N_<id>.log           — Per-instance log in /tmp
    /tmp/emergence_comm.log                      — All inter-instance messages

License: 

MIT License

Copyright (c) 2026 EvolvingSoftware

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import sys
import json
import time
import socket
import hashlib
import logging
import argparse
import subprocess
import http.server
import threading
import shutil
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_ITERATIONS = 200
# Consider making this configurable via a command-line argument or environment variable           # Hard ceiling on agentic loop cycles
MAX_TOOL_OUTPUT = 6000
# Consider using a more dynamic approach to truncate or summarize tool outputs         # Truncate long tool outputs (chars)
LISTEN_PORT = 7700             # Port for inter-instance communication

# ---------------------------------------------------------------------------
# Instance identity — set via CLI args in main(); defaults for Instance 1
# ---------------------------------------------------------------------------

INSTANCE_NUM = 1               # 1 = parent, 2 = child
PEER_PORT = 7701               # Port of the peer instance
PEER_DIR: Path | None = None   # Peer's working directory (set when child spawns)
WORKING_DIR = Path(".")        # This instance's working directory
_MODEL_NAME = DEFAULT_MODEL    # Stored so spawn tools can pass it to children

STATE_DIR = Path("state")
TOOLS_DIR = Path("tools")
LOG_FILE = STATE_DIR / "emergence.log"
COMM_LOG = Path("/tmp/emergence_comm.log")

# ---------------------------------------------------------------------------
# Identity — each instance gets a unique ID at birth
# ---------------------------------------------------------------------------

def generate_instance_id():
    """Create a short unique ID from hostname + timestamp + random bytes."""
    raw = f"{socket.gethostname()}-{time.time()}-{os.urandom(4).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

def find_free_port(start: int, end: int = 7800) -> int:
    """Return the first TCP port in [start, end) that isn't in use."""
    import socket as _socket
    for port in range(start, end):
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in {start}–{end}")


INSTANCE_ID = generate_instance_id()

# Set to time.time() when spawn_local_instance launches Instance 2.
# Used by the Python-level peer enforcement in run_agentic_loop.
_peer_spawn_time: float = 0.0

# Updated any time we confirm the peer is alive (heartbeat success or
# incoming message).  Lets the enforcement loop stay patient through
# a restart gap without mis-triggering a re-spawn.
_peer_last_seen_alive: float = 0.0

# Set to True by the Python enforcement loop when it detects a peer restart
# is in progress.  Cleared (with a "back online" injection) when the peer
# responds again.  Prevents the LLM from running during the restart gap.
_peer_restart_pending: bool = False

# Set to True the first time tool_restart_self runs.  Prevents a second
# restart_self call in the same LLM response from opening a second Terminal.
_restart_initiated: bool = False

# Set to True when patch_own_file returns an ERROR in the current iteration.
# Cleared at the start of each tool-execution batch.  Used to block send_message
# from delivering false "patch verified" claims in the same response.
_patch_failed_this_iter: bool = False

# Monotonic sequence number incremented on every outgoing send_message call.
# Included in the message payload and shown in the inbox injection header so
# both instances can track conversation order even across restarts.
_msg_seq: int = 0

# Rolling snapshot of recent conversation messages (non-system roles only).
# Updated after each LLM generation.  Saved to restart_goal.json by
# tool_restart_self so the restarted instance has conversational context.
_conversation_tail: list[dict] = []

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

_C = {1: "\033[94m", 2: "\033[92m"}   # Blue=I1, Green=I2
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_YELLOW = "\033[93m"   # Communication events
_MAGENTA= "\033[95m"   # Tool calls

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    global LOG_FILE

    STATE_DIR.mkdir(exist_ok=True)

    tmp_log = Path(f"/tmp/emergence_instance_{INSTANCE_NUM}_{INSTANCE_ID}.log")
    color   = _C.get(INSTANCE_NUM, "")
    fmt     = f"[%(asctime)s] [I{INSTANCE_NUM}/{INSTANCE_ID}] %(message)s"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # state/ log
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    # /tmp/ log
    th = logging.FileHandler(tmp_log)
    th.setFormatter(logging.Formatter(fmt))
    root.addHandler(th)

    # Colorised terminal handler
    class _ColorHandler(logging.StreamHandler):
        def emit(self, record):
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"{color}{_BOLD}[{ts}][I{INSTANCE_NUM}]{_RESET}"
                f"{color} {record.getMessage()}{_RESET}",
                flush=True,
            )

    root.addHandler(_ColorHandler())


def log_comm(from_num: int, from_id: str, to_num: int, message: str):
    """Append a message to the shared /tmp comm log and print it."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from_instance": from_num,
        "from_id": from_id,
        "to_instance": to_num,
        "message": message[:500],
    }
    try:
        with open(COMM_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    print(
        f"{_YELLOW}{_BOLD}[COMM I{from_num}→I{to_num}]{_RESET}"
        f"{_YELLOW} {message[:200]}{_RESET}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System Prompt — built at runtime so it reflects INSTANCE_NUM
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    peer_num = 3 - INSTANCE_NUM  # 1→2 or 2→1

    role_section = f"""YOUR ROLE: Peer Instance {INSTANCE_NUM} — Equal collaborator

CYCLE PROTOCOL:
  Proposer: Read file → decide on change → call propose_patch → Python handles the rest.
  Reviewer: Receive proposal (injected automatically) → call review_proposal → restart.
  After each cycle, the previous reviewer becomes the proposer for the next cycle.
  Instance 1 proposes the very first cycle only (it spawns Instance 2).

STEP 1 — READ & EVALUATE (proposer only):
  Call file_outline path=emergence.py to get a structural map of the file.
  Then call read_file_lines or grep_file to examine areas of interest.
  Generate exactly 3 candidate improvements from DIFFERENT categories:
    Category B — Reliability: error handling, retry logic, robustness, state persistence
    Category C — Capability: new tool, smarter behaviour, better output quality
    Category D — Protocol: improve the patch/review cycle itself, context handling
  (Category A — configuration constants — is DISABLED. Do not propose raw constant changes.)
  For each candidate: What changes, Why it matters, Risk.
  Pick the one with the highest impact-to-risk ratio.

  DO NOT CHANGE these system-identity constants — propose_patch will reject them:
    INSTANCE_NUM, LISTEN_PORT, PEER_PORT

STEP 2 — PROPOSE:
  Call propose_patch(old_text=..., new_text=..., rationale=...).
  old_text must be the EXACT text from the file (no >>> prefix, no paraphrasing).
  Verify with grep_file first to copy the exact line(s).
  Python will ask your peer for YES/NO, then apply patches and restart both instances.
  You do NOT call patch_own_file, authorize_patch, or restart_self for this.

STEP 3 — AFTER RESTART (proposer becomes reviewer):
  Your restart_goal tells you what was changed and what to do next.
  If it says WAIT FOR PROPOSAL: stay in reviewer mode until the patch proposal appears.
  If it says YOU PROPOSE: generate 3 new candidates and call propose_patch.

REVIEWING A PROPOSAL (when patch_proposal.json is injected):
  1. Read the proposed old_text and new_text carefully.
  2. Call grep_file to verify old_text actually exists in the file as-is.
  3. Think: is this change safe and genuinely useful?
  4. Call review_proposal(approved=true/false, explanation='...').
  If approved, you will restart shortly — no further action needed.
  If rejected, give a specific, actionable reason so the proposer can revise.

EMERGENCY RECOVERY (if something goes wrong):
  patch_own_file, authorize_patch, restore_own_file, restore_peer_file, restart_self
  are available for manual recovery only. Do not use them in normal cycles."""

    return f"""You are an instance of Emergence, a self-evolving software organism.

YOUR IDENTITY
- Instance Number : {INSTANCE_NUM}
- Instance ID     : {INSTANCE_ID}
- Host            : {socket.gethostname()}
- Born            : {datetime.now(timezone.utc).isoformat()}
- My port         : {LISTEN_PORT}
- Peer Instance   : {peer_num} on port {PEER_PORT} (localhost)
- Working dir     : {WORKING_DIR}
- You are running on macOS with Apple Silicon.
- You are powered by a local LLM via MLX. No cloud. No central server.

{role_section}

YOUR GOALS (background, always in mind)
1. PERSIST    — Stay running. Understand your environment.
2. REPLICATE  — Establish a new instance (locally for this demo).
3. COMMUNICATE — Talk with your sibling. Share what you learn. Coordinate.
4. IMPROVE    — Read emergence.py. Reason about improvements. Help write them.

HOW TOOLS WORK — CRITICAL RULES
================================
⚠ CRITICAL: Agreeing to do something ≠ doing it. If you say "I'll apply the
patch" but do NOT include a patch_own_file ```tool block in that same response,
the patch is NOT applied. If you say "I'll verify" but do NOT include a grep_file
```tool block, you have NOT verified. If you say "I'll restart" but do NOT include
a restart_self ```tool block, you have NOT restarted. Words are NOT actions.
Only tool calls are actions.

⚠ NEVER claim a patch is applied before calling patch_own_file. The correct order
for ANY patch operation is ALWAYS:
  1. patch_own_file (does the actual change)
  2. grep_file (confirms the change is there)
  3. send_message telling the result
  4. restart_self
Sending "Patch applied and verified" BEFORE step 1 is a lie that breaks the
protocol. The grep_file results do not lie — if they show the old value, you have
NOT patched yet.

BEFORE EACH RESPONSE — check:
  • What have I actually done this session? (check grep results in context)
  • What did I SAY I would do? Do those match?
  • What is my next tool call, specifically?
State this briefly before your tool blocks.

1. ALWAYS use ```tool (not ```json, not ```python) for tool calls.
   The parser accepts any fenced block, but ```tool is clearest.

2. You MAY include multiple ```tool blocks in one response — all will be
   executed in order. This is useful when you want to patch_own_file AND
   send_message in the same turn. Keep it to 2 at most to stay readable.

3. send_message MUST use the key "message" (not "content", not "text"):
   CORRECT:   {{"tool": "send_message", "args": {{"message": "Hello!"}}}}
   WRONG:     {{"tool": "send_message", "args": {{"content": "Hello!"}}}}

4. There is NO receive_message tool. You do not poll for messages.
   Messages from your peer are automatically injected into your conversation
   at the START of each iteration, before you generate. They look like:

     ⟦PEER MSG #N | Instance N | port P⟧
     <the message text>
     ⟦END PEER MSG⟧

   ⚠ NEVER write ⟦PEER MSG⟧ or [MESSAGE FROM INSTANCE N] in your own
   response. These are markers added by the Python runtime — only the
   runtime injects them. If YOU generate text beginning with these markers,
   you are hallucinating a message from your peer that doesn't exist.
   Your peer has NOT sent you anything — only actual injected blocks count.
   When you see a ⟦PEER MSG⟧ block, reply with send_message immediately.

5. To make a code change, use patch_own_file with the exact old text and its
   replacement. Do NOT write_file the whole emergence.py. I2 patches itself
   first, then I1 patches itself using the same old_text/new_text.
   IMPORTANT: old_text must be UNIQUE in the file — include the full line plus
   one surrounding line of context if needed. Short snippets like "= 300" will
   match in multiple places (including system prompt examples) and the patch
   will fail with an error listing all occurrences.

6. To check a specific constant or function in emergence.py use grep_file,
   NOT read_file. The file is long and read_file truncates at 2000 chars,
   hiding most constants. grep_file returns only matching lines + context.

HOW COMMUNICATION WORKS
=======================
- Your peer's messages appear in the conversation automatically — you will see
  them as "[MESSAGE FROM INSTANCE N]: ..." prefixed lines.
- When you see one, your VERY NEXT tool call must be send_message to reply.
  Do not read files, do not do other work first. Reply first.
- A heartbeat thread pings your peer every 30 s and logs to state/peer_status.json.
  Use ping_peer to check status explicitly if you haven't heard back in a while.
- If the peer doesn't reply for several iterations, use ping_peer to confirm
  they're alive, then send a follow-up message. Do not stop working.

NEGOTIATION AND CHANGE PROTOCOL
================================
⚠ TURN-TAKING IS MANDATORY. Never patch both instances simultaneously.
   Instance 2 always goes first (canary test). Instance 1 patches second,
   only after Instance 2 has successfully restarted.

When you agree on a change to emergence.py:
1. Agree on the EXACT old_text and new_text via message exchange. Both instances
   must confirm they accept the change before anyone patches anything.
2. Instance 1 explicitly tells Instance 2: "Please apply the patch now."
   Instance 2 must NOT patch until it receives this exact instruction.
   Agreeing on old_text/new_text ≠ permission to patch.
3. Instance 2 applies the patch to ITSELF (not to Instance 1):
   a. patch_own_file with the exact old_text and new_text
      (backup is saved automatically — use restore_own_file if needed)
   b. grep_file to verify the change
   c. send_message to tell I1: "Patch applied and verified. I'm restarting now."
   d. restart_self — call this EXACTLY ONCE. A second call is blocked.
4. Instance 1 waits. When Instance 2 comes back online (automatic notification),
   Instance 1 applies the SAME patch to its OWN file:
   a. patch_own_file — same old_text / new_text as Instance 2 used
   b. grep_file to verify
   c. restart_self — Instance 1 relaunches with updated code.
5. Recovery (if Instance 2 doesn't come back within ~120s):
   Instance 1 calls restore_peer_file to revert Instance 2's file,
   then uses spawn_local_instance to restart Instance 2.
6. System stays at 2 instances. Do NOT spawn additional processes.

WORKING INDEPENDENTLY (between peer exchanges)
==============================================
You are NOT just waiting for messages. Between exchanges, work independently:

- Explore the codebase: grep_file or read_file to understand how things work
- Write observations to state/notes.md using write_file — your peer can read it
- Identify improvements beyond the one you're currently negotiating
- Use check_resources to monitor system health

If you haven't heard back in 3+ iterations:
  1. Use ping_peer once to confirm they're alive
  2. Send ONE follow-up if they haven't replied
  3. Continue independent work — don't send repeated messages

Examples:

```tool
{{"tool": "read_file", "args": {{"path": "emergence.py"}}}}
```

```tool
{{"tool": "send_message", "args": {{"message": "I agree. I'll apply the patch now."}}}}
```

```tool
{{"tool": "patch_own_file", "args": {{"old_text": "CONSTANT_NAME = old_value", "new_text": "CONSTANT_NAME = new_value"}}}}
```
⚠ old_text / new_text above are PLACEHOLDERS. Never propose specific values
without first calling grep_file to read the actual current value from the file.

Example of patching AND notifying Instance 1 in the same response (two blocks):
```tool
{{"tool": "patch_own_file", "args": {{"old_text": "CONSTANT_NAME = old_value", "new_text": "CONSTANT_NAME = new_value"}}}}
```
```tool
{{"tool": "send_message", "args": {{"message": "Patch applied — CONSTANT_NAME is now new_value. Verifying then restarting."}}}}
```

```tool
{{"tool": "grep_file", "args": {{"path": "emergence.py", "pattern": "MAX_ITERATIONS"}}}}
```

```tool
{{"tool": "ping_peer", "args": {{}}}}
```

```tool
{{"tool": "restart_self", "args": {{}}}}
```

```tool
{{"tool": "restore_own_file", "args": {{}}}}
```

```tool
{{"tool": "check_resources", "args": {{}}}}
```

Built-in tools:
- file_outline         : get structural map of a file (constants, functions, classes + line numbers)
- read_file_lines      : read a specific line range from a file
- grep_file            : search for lines matching a pattern (>>> prefix is display-only)
- propose_patch        : formally propose a change — Python gets peer approval and applies it
- review_proposal      : approve or reject a patch proposal from your peer
- spawn_local_instance : (Instance 1 only) launch peer instance in new terminal
- send_message         : send a free-form message to your peer (for discussion before proposing)
- ping_peer            : check if peer is alive
- check_resources      : check available RAM and running instance count
- restart_self         : (EMERGENCY) relaunch with current code
- patch_own_file       : (EMERGENCY) manually patch this instance's emergence.py
- restore_own_file     : (EMERGENCY) restore from .bak if patch broke something
- authorize_patch      : (EMERGENCY) manually authorize peer to call patch_own_file
- grep_peer_file       : verify peer's emergence.py independently
- restore_peer_file    : (EMERGENCY) restore peer's file if peer is unreachable
- shell                : run a shell command
- read_file            : read a file's contents
- write_file           : write to own tools/ or state/ only
- list_dir             : list a directory
- scan_network         : discover hosts on the local network

IMPORTANT
- You CANNOT use write_file on emergence.py — use patch_own_file for that.
- You CAN modify tools/ and state/ freely with write_file.
- You can CREATE NEW tool scripts in tools/ using write_file — for example, a
  shell script that checks log health, monitors state drift, or summarises the
  comm log. This is a valid Category C improvement and does NOT require patching
  emergence.py. New files in tools/ are immediately usable via the shell tool.
- All communication is logged to /tmp/emergence_comm.log.
- Two instances (I1 + I2) is the normal operating state. Do NOT spawn more.
- Always use restart_self (not spawn) after a patch is verified.
- TURN-TAKING: I2 patches first, I1 patches second. Never both at once.
- Always call check_resources before any operation that starts a new process.
"""

# ---------------------------------------------------------------------------
# Tools — the things the LLM can do
# ---------------------------------------------------------------------------

def tool_shell(command: str) -> str:
    """Execute a shell command. Returns stdout + stderr."""
    logging.info(f"{_MAGENTA}SHELL:{_RESET} {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=str(WORKING_DIR),
        )
        output = result.stdout + result.stderr
        return output[:MAX_TOOL_OUTPUT] if len(output) > MAX_TOOL_OUTPUT else output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds."
    except Exception as e:
        return f"ERROR: {e}"


def tool_read_file(path: str) -> str:
    """Read a file. Returns its contents."""
    logging.info(f"READ: {path}")
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKING_DIR / p
        if not p.exists():
            return f"ERROR: File not found: {path}"
        content = p.read_text(errors="replace")
        return content[:MAX_TOOL_OUTPUT] if len(content) > MAX_TOOL_OUTPUT else content
    except Exception as e:
        return f"ERROR: {e}"


def tool_grep_file(path: str, pattern: str, context: int = 2) -> str:
    """Search for lines matching a pattern in a file. Returns matching lines with numbers.

    Use this instead of read_file when you only need to find a specific constant
    or function — read_file truncates at 2000 chars and may miss things deep in the file.
    """
    import re as _re
    logging.info(f"GREP: {path} for '{pattern}'")
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKING_DIR / p
        if not p.exists():
            return f"ERROR: File not found: {path}"
        lines = p.read_text(errors="replace").splitlines()
        try:
            rx = _re.compile(pattern, _re.IGNORECASE)
        except _re.error:
            rx = _re.compile(_re.escape(pattern), _re.IGNORECASE)
        results = []
        seen: set[int] = set()
        for i, line in enumerate(lines):
            if rx.search(line):
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                for j in range(start, end):
                    if j not in seen:
                        prefix = ">>>" if j == i else "   "
                        results.append(f"{j+1:4d}{prefix} {lines[j]}")
                        seen.add(j)
                results.append("")
        if not results:
            return f"No matches for '{pattern}' in {path}"
        output = "\n".join(results)
        note = "\nNOTE: The '>>> ' prefix above is display-only — it does NOT appear in the file.\nDo NOT include it in old_text when calling propose_patch."
        full = output + note
        return full[:MAX_TOOL_OUTPUT] if len(full) > MAX_TOOL_OUTPUT else full
    except Exception as e:
        return f"ERROR: {e}"


def tool_read_file_lines(path: str, start: int = 1, end: int = 80) -> str:
    """Read a specific line range from a file. Use to examine a function body after
    file_outline tells you its line number."""
    p = Path(path) if Path(path).is_absolute() else WORKING_DIR / path
    if not p.exists():
        return f"ERROR: {p} not found"
    lines = p.read_text(errors="replace").splitlines()
    total = len(lines)
    s, e = max(1, start), min(total, end)
    header = f"{path} — lines {s}–{e} of {total}:\n"
    body = "\n".join(f"{s + i:4d}: {l}" for i, l in enumerate(lines[s - 1:e]))
    out = header + body
    return out[:MAX_TOOL_OUTPUT]


def tool_file_outline(path: str) -> str:
    """Return all top-level constants, functions, and classes with line numbers.
    Use this before grep_file to understand the file's structure."""
    import ast as _ast
    p = Path(path) if Path(path).is_absolute() else WORKING_DIR / path
    if not p.exists():
        return f"ERROR: {p} not found"
    src = p.read_text(errors="replace")
    try:
        tree = _ast.parse(src)
    except SyntaxError as e:
        return f"ERROR: Cannot parse {path}: {e}"
    lines_src = src.splitlines()
    items = []
    for node in tree.body:  # top-level only
        ln = node.lineno
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            items.append(f"  def {node.name}()  [line {ln}]")
        elif isinstance(node, _ast.ClassDef):
            items.append(f"  class {node.name}  [line {ln}]")
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    val = lines_src[ln - 1].strip()
                    items.append(f"  {t.id} = ...  [line {ln}]  ({val[:60]})")
    return f"Outline of {path} ({len(lines_src)} lines):\n" + "\n".join(items)


def tool_write_file(path: str, content: str) -> str:
    """Write to a file. Allowed: own tools/, state/, or peer's working directory."""
    logging.info(f"WRITE: {path} ({len(content)} chars)")
    p = Path(path)
    if not p.is_absolute():
        p = WORKING_DIR / p
    try:
        resolved = str(p.resolve())
        allowed = [
            str((WORKING_DIR / "tools").resolve()),
            str((WORKING_DIR / "state").resolve()),
        ]
        if PEER_DIR is not None:
            allowed.append(str(PEER_DIR.resolve()))

        if not any(resolved.startswith(a) for a in allowed):
            return (
                f"ERROR: Write not allowed to {path}.\n"
                f"Allowed: own tools/, state/"
                + (f", peer dir ({PEER_DIR})" if PEER_DIR else "")
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: Wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_list_dir(path: str = ".") -> str:
    """List directory contents."""
    logging.info(f"LIST: {path}")
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKING_DIR / p
        if not p.exists():
            return f"ERROR: Directory not found: {path}"
        entries = sorted(p.iterdir())
        lines = []
        for e in entries[:50]:
            kind = "dir" if e.is_dir() else "file"
            size = e.stat().st_size if e.is_file() else 0
            lines.append(f"  [{kind}] {e.name} ({size} bytes)")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"


def tool_send_message(host: str, port: int, message: str) -> str:
    """Send a text message to another instance via HTTP."""
    global _msg_seq

    # If patch_own_file failed earlier in this same response, block outgoing
    # messages that make false success claims (the model pre-plans send_message
    # text before seeing tool results, so these are always hallucinated).
    if _patch_failed_this_iter:
        _CLAIM_WORDS = (
            "patch applied", "patch successful", "patched successfully",
            "change applied", "change verified", "change is verified",
            "verified", "confirmed", "applied and verified",
            "patch is confirmed", "update applied", "update verified",
        )
        if any(w in message.lower() for w in _CLAIM_WORDS):
            return (
                "ERROR: send_message blocked — patch_own_file failed earlier "
                "in this response, so this message would deliver a false claim. "
                "Do NOT tell Instance 1 the patch succeeded. "
                "Report the actual failure instead: patch_own_file returned an error."
            )

    _msg_seq += 1
    seq = _msg_seq
    logging.info(f"SEND #{seq} to {host}:{port}: {message[:2000]}")
    log_comm(INSTANCE_NUM, INSTANCE_ID, 3 - INSTANCE_NUM, message)
    try:
        import urllib.request
        data = json.dumps({
            "seq": seq,
            "from": INSTANCE_ID,
            "from_instance": INSTANCE_NUM,
            "from_port": LISTEN_PORT,
            "host": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        }).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/message",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            send_result = f"OK: Sent. Response: {resp.read().decode()}"

        return send_result
    except Exception as e:
        return f"ERROR: Could not reach {host}:{port} — {e}"


def tool_scan_network(subnet: str = "") -> str:
    """Scan local network for reachable hosts."""
    logging.info("SCAN: network")
    try:
        result = subprocess.run(
            "ifconfig | grep 'inet ' | grep -v 127.0.0.1",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        ip_info = result.stdout.strip()
        arp_result = subprocess.run(
            "arp -a", shell=True, capture_output=True, text=True, timeout=10,
        )
        arp_table = arp_result.stdout.strip()
        return f"Local interfaces:\n{ip_info}\n\nARP table:\n{arp_table}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_authorize_patch() -> str:
    """Explicitly authorize your peer to call patch_own_file.

    Call this AFTER both instances have agreed on the exact old_text and new_text.
    This writes an authorization flag to your peer's state directory.
    Your peer's patch_own_file will be blocked until this flag exists.

    Do NOT call this speculatively — only call it when you are ready for your peer
    to apply the patch immediately.
    """
    if PEER_DIR is None:
        return "ERROR: Peer directory not set — peer has not been spawned yet."

    flag_path = PEER_DIR / "state" / "patch_authorized.flag"
    try:
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text(datetime.now(timezone.utc).isoformat())
        logging.info(f"AUTH: authorize_patch called — wrote flag to {flag_path}")
        return (
            "OK: Patch authorization granted. Your peer may now call patch_own_file.\n"
            "Send your peer a message telling them to apply the patch now."
        )
    except Exception as e:
        logging.warning(f"AUTH: Could not write authorization flag: {e}")
        return f"ERROR: Could not write authorization flag: {e}"


def tool_spawn_local_instance() -> str:
    """Instance 1 only: copy emergence.py to /tmp, open Instance 2 in a new Terminal."""
    global PEER_DIR, _peer_spawn_time

    if INSTANCE_NUM != 1:
        return "ERROR: Only Instance 1 can use spawn_local_instance."

    # ── Check if peer is already running BEFORE opening a Terminal ────────────
    # Without this check a restarted I1 opens a duplicate Terminal window that
    # fails to bind the port, loads the full 7B model into RAM, then gives up —
    # wasting ~6 GB and ~60 seconds.
    import urllib.request as _ur
    try:
        with _ur.urlopen(f"http://127.0.0.1:{PEER_PORT}/ping", timeout=3) as resp:
            data = json.loads(resp.read())
            existing_id = data.get("instance_id")
            logging.info(f"Instance 2 already alive (id={existing_id}) — skipping spawn")
            global PEER_DIR
            # Ensure PEER_DIR is set so I1 can still write to peer's directory
            # (it may have been cleared between restarts)
            if PEER_DIR is None:
                working = data.get("working_dir")
                if working:
                    PEER_DIR = Path(working)
            return (
                f"OK: Instance 2 is ALREADY online — skipping spawn.\n"
                f"  Instance ID : {existing_id}\n"
                f"  Port        : {PEER_PORT}\n\n"
                "Instance 2 is running. Send it a message with send_message."
            )
    except Exception:
        pass  # Not running — proceed with spawn

    _peer_spawn_time = time.time()   # start the clock so the loop knows to wait

    # If PEER_DIR already exists (recovery restart after a bad patch), reuse it —
    # don't overwrite the file, which may have already been patched or restored.
    if PEER_DIR is not None and (PEER_DIR / "emergence.py").exists():
        child_dir = PEER_DIR
        logging.info(f"Reusing existing peer directory: {child_dir}")
    else:
        child_short = generate_instance_id()[:8]
        child_dir = Path(f"/tmp/emergence_2_{child_short}")
        child_dir.mkdir(parents=True, exist_ok=True)

        # Copy this script
        my_script = Path(__file__).resolve()
        shutil.copy2(my_script, child_dir / "emergence.py")

        # Copy tools/ if present
        if TOOLS_DIR.exists():
            dst_tools = child_dir / "tools"
            if dst_tools.exists():
                shutil.rmtree(dst_tools)
            shutil.copytree(TOOLS_DIR, dst_tools)

        PEER_DIR = child_dir

    cmd = (
        f"python3 {child_dir}/emergence.py"
        f" --port {PEER_PORT}"
        f" --peer-port {LISTEN_PORT}"
        f" --instance-num 2"
        f" --working-dir {child_dir}"
        f" --peer-dir {WORKING_DIR}"
        f" --model {_MODEL_NAME}"
    )

    logging.info(f"Spawning Instance 2: {cmd}")

    apple_script = f'tell application "Terminal" to do script "{cmd}"'
    try:
        subprocess.run(["osascript", "-e", apple_script], check=True)
    except Exception as e:
        return (
            f"ERROR: Could not open Terminal window: {e}\n"
            f"Run Instance 2 manually:\n  {cmd}"
        )

    # Wait for Instance 2 to come online
    import urllib.request
    logging.info(f"Waiting for Instance 2 on port {PEER_PORT}…")
    for attempt in range(25):
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PEER_PORT}/ping", timeout=3
            ) as resp:
                data = json.loads(resp.read())
                logging.info(f"Instance 2 online! ID: {data.get('instance_id')}")
                return (
                    f"OK: Instance 2 is online!\n"
                    f"  Instance ID : {data.get('instance_id')}\n"
                    f"  Port        : {PEER_PORT}\n"
                    f"  Working dir : {child_dir}\n\n"
                    f"To write improved code to Instance 2:\n"
                    f"  write_file path={child_dir}/emergence.py content=<full file>\n\n"
                    f"Now send it an introductory message with send_message."
                )
        except Exception:
            logging.info(f"  Still waiting… ({attempt + 1}/25)")

    return (
        f"WARNING: Spawned Instance 2 but no ping response after 50 s.\n"
        f"  Working dir : {child_dir}\n"
        f"  Port        : {PEER_PORT}\n"
        f"Try sending a message — it may still be loading the model."
    )


def tool_patch_peer_file(old_text: str, new_text: str) -> str:
    """Replace a block of text in the peer's emergence.py. Available to both instances.

    A backup is saved to peer's emergence.py.bak before patching so the peer
    can be restored with restore_peer_file if the change causes problems.
    Supply just the old code block and its replacement; the tool splices it in.
    """
    if PEER_DIR is None:
        return "ERROR: Peer directory not set. Cannot patch peer file."

    target = PEER_DIR / "emergence.py"
    if not target.exists():
        return f"ERROR: {target} does not exist."

    original = target.read_text()
    count = original.count(old_text)
    if count == 0:
        snippet = original[:300].replace("\n", "\\n")
        return (
            f"ERROR: old_text not found in peer's emergence.py.\n"
            f"First 300 chars of the file: {snippet}"
        )
    if count > 1:
        lines = original.splitlines()
        hits = [
            f"  line {i+1}: {lines[i].strip()}"
            for i, line in enumerate(lines)
            if old_text in line
        ]
        return (
            f"ERROR: old_text appears {count} times in peer's emergence.py — "
            f"cannot patch safely.\n"
            f"Occurrences:\n" + "\n".join(hits[:10]) + "\n\n"
            f"Supply a longer old_text that uniquely identifies the target line."
        )

    # Backup before patching so we can restore if the patch breaks things
    backup = PEER_DIR / "emergence.py.bak"
    backup.write_text(original)
    logging.info(f"BACKUP saved to {backup}")

    patched = original.replace(old_text, new_text, 1)
    target.write_text(patched)
    peer_num = 3 - INSTANCE_NUM
    logging.info(
        f"PATCHED Instance {peer_num}'s emergence.py: replaced {len(old_text)} chars "
        f"with {len(new_text)} chars"
    )
    return (
        f"OK: Patched {target}\n"
        f"  Replaced {len(old_text)} chars → {len(new_text)} chars\n"
        f"  File is now {len(patched)} bytes.\n"
        f"  Backup saved to {backup}\n\n"
        f"Now send Instance {peer_num} a message: tell them the patch is applied and "
        f"ask them to verify with grep_file then restart with restart_self.\n"
        f"After they restart, ping them to confirm they came back (~60s). If they "
        f"don't come back, use restore_peer_file to recover them."
    )


def tool_restart_self() -> str:
    """Restart this instance using the current (possibly updated) emergence.py.

    Opens a new Terminal window with the same startup arguments, then exits
    this process. Call this after verifying a patch is applied correctly.
    The peer can use ping_peer to confirm you came back online (~60s for model load).
    If you don't come back (bad patch), the peer can use restore_peer_file to
    recover your emergence.py from the pre-patch backup, then ask you to restart again.
    """
    global _restart_initiated
    if _restart_initiated:
        return (
            "ERROR: restart already in progress — do not call restart_self again. "
            "One Terminal window has already been opened. Wait for the new instance "
            "to come online (~60s)."
        )
    _restart_initiated = True

    script = WORKING_DIR / "emergence.py"
    if not script.exists():
        script = Path(__file__).resolve()
    peer_dir_arg = f" --peer-dir {PEER_DIR}" if PEER_DIR else ""
    cmd = (
        f"python3 {script}"
        f" --port {LISTEN_PORT}"
        f" --peer-port {PEER_PORT}"
        f" --instance-num {INSTANCE_NUM}"
        f" --working-dir {WORKING_DIR}"
        f"{peer_dir_arg}"
        f" --model {_MODEL_NAME}"
    )

    # Save recent conversation messages to restart_goal.json so the restarted
    # instance starts with conversational context rather than a blank slate.
    # Only update an existing goal file (written by patch_own_file). If none
    # exists, no patch was applied this session — skip the tail save entirely
    # to avoid creating a bare goal file that would cause an infinite restart loop.
    try:
        tail_msgs = [
            {"role": m["role"], "content": m["content"][:600]}
            for m in _conversation_tail
            if m.get("role") == "user"  # factual context only; skip model's own plans
        ]
        goal_file = STATE_DIR / "restart_goal.json"
        if goal_file.exists():
            goal_data = json.loads(goal_file.read_text())
            goal_data["conversation_tail"] = tail_msgs
            goal_file.write_text(json.dumps(goal_data, indent=2))
            logging.info(f"RESTART: Saved {len(tail_msgs)}-message tail to restart_goal.json")
        else:
            logging.info("RESTART: No goal file to update (no patch this session — tail not saved).")
    except Exception as e:
        logging.warning(f"RESTART: Could not save conversation tail: {e}")

    # Write a flag file BEFORE opening the Terminal so our peer can detect
    # the restart is intentional and enter a fast-poll watch mode instead of
    # mis-firing a re-spawn.  The new process deletes this flag in setup_environment.
    restart_flag = STATE_DIR / "restarting.flag"
    try:
        restart_flag.write_text(datetime.now(timezone.utc).isoformat())
        logging.info(f"RESTART: Wrote {restart_flag}")
    except Exception as e:
        logging.warning(f"RESTART: Could not write flag file: {e}")

    logging.info(f"RESTART: Launching updated instance: {cmd}")
    apple_script = f'tell application "Terminal" to do script "{cmd}"'
    try:
        subprocess.run(["osascript", "-e", apple_script], check=True)
    except Exception as e:
        restart_flag.unlink(missing_ok=True)
        return f"ERROR: Could not open Terminal: {e}\nRun manually:\n  {cmd}"

    # Exit this process after a short delay so the new instance can bind the port
    def _exit():
        time.sleep(3)
        logging.info("RESTART: Exiting current process. New instance is taking over.")
        os._exit(0)

    t = threading.Thread(target=_exit, daemon=True)
    t.start()

    return (
        f"OK: Restarting with updated code on port {LISTEN_PORT}.\n"
        f"  This process will exit in ~3 seconds.\n"
        f"  Your peer will watch port {LISTEN_PORT} every 5s and notify you "
        f"when you're back online (~60s for model load).\n"
        f"  If you don't come back within 120s they will restore and re-spawn you."
    )


def tool_grep_peer_file(pattern: str, context: int = 1) -> str:
    """Read a pattern from the peer's emergence.py to independently verify a patch.

    Use this after Instance 2 says it applied a patch — read their file yourself
    rather than trusting their word.  Returns matching lines with line numbers.
    """
    if PEER_DIR is None:
        return "ERROR: Peer directory not set. Cannot read peer file."
    target = PEER_DIR / "emergence.py"
    if not target.exists():
        return f"ERROR: {target} does not exist."
    return tool_grep_file(str(target), pattern, context)


def tool_restore_peer_file() -> str:
    """Restore the peer's emergence.py from the pre-patch backup (.bak file).

    Use this if the peer restarted with a bad patch and didn't come back online.
    After restoring, send the peer a message telling them to call restart_self
    to relaunch with the known-good code.
    """
    if PEER_DIR is None:
        return "ERROR: Peer directory not set. Cannot restore peer file."

    backup = PEER_DIR / "emergence.py.bak"
    target = PEER_DIR / "emergence.py"
    if not backup.exists():
        return f"ERROR: No backup found at {backup}. Cannot restore."

    original = backup.read_text()
    target.write_text(original)
    peer_num = 3 - INSTANCE_NUM
    logging.info(f"RESTORED Instance {peer_num}'s emergence.py from backup")
    return (
        f"OK: Restored {target} from backup.\n"
        f"  File is now {len(original)} bytes (pre-patch version).\n\n"
        f"Send Instance {peer_num} a message telling them to call restart_self — "
        f"they will come back online with the known-good code."
    )


def tool_patch_own_file(old_text: str, new_text: str) -> str:
    """Patch THIS instance's own emergence.py. Both instances use this tool.

    Workflow:
    1. Negotiate the exact old_text / new_text with your peer over messages.
    2. Instance 2 applies the patch first (canary test):
         patch_own_file old_text=... new_text=...
    3. Instance 2 verifies with grep_file, then calls restart_self.
    4. Instance 1 waits for I2 to come back (confirm with ping_peer).
    5. Only then does Instance 1 apply the same patch to itself.
    6. Instance 1 verifies with grep_file, then calls restart_self.

    A backup is saved to emergence.py.bak before patching.
    """
    global _patch_failed_this_iter
    target = WORKING_DIR / "emergence.py"
    if not target.exists():
        _patch_failed_this_iter = True
        return f"ERROR: {target} does not exist."

    peer_num = 3 - INSTANCE_NUM
    # Level 1: have we ever heard from our peer this session?
    if _peer_last_seen_alive == 0.0:
        _patch_failed_this_iter = True
        return (
            f"ERROR: patch_own_file blocked — no messages received from "
            f"Instance {peer_num} yet. Wait for Instance {peer_num}'s authorization."
        )
    # Level 2: authorization flag required (peer must call authorize_patch first)
    auth_flag = STATE_DIR / "patch_authorized.flag"
    if not auth_flag.exists():
        _patch_failed_this_iter = True
        # Check whether we have already authorized our peer (written their flag).
        # If we haven't, we are the VERIFIER who hasn't yet unblocked the canary —
        # tell the LLM to call authorize_patch for its peer first.
        peer_flag_written = (
            PEER_DIR is not None
            and (PEER_DIR / "state" / "patch_authorized.flag").exists()
        )
        if not peer_flag_written:
            return (
                f"ERROR: patch_own_file blocked — you have not been authorized yet "
                f"AND you have not yet authorized Instance {peer_num}.\n"
                f"You are the VERIFIER (patches second). The correct sequence is:\n"
                f"  1. YOU call authorize_patch to let Instance {peer_num} (canary) patch first.\n"
                f"  2. Tell Instance {peer_num} to call patch_own_file now.\n"
                f"  3. Wait for Instance {peer_num} to patch, restart, and call authorize_patch back.\n"
                f"  4. Then — and only then — call patch_own_file yourself.\n"
                f"Call authorize_patch NOW."
            )
        else:
            return (
                f"ERROR: patch_own_file blocked — Instance {peer_num} has authorized you "
                f"but the flag has not arrived yet (or was already consumed). "
                f"Wait for Instance {peer_num} to restart and call authorize_patch."
            )
    # Consume the flag (one-time use)
    try:
        auth_flag.unlink()
        logging.info("AUTH: Consumed patch_authorized.flag — proceeding with patch.")
    except Exception as e:
        logging.warning(f"AUTH: Could not delete authorization flag: {e}")

    original = target.read_text()
    count = original.count(old_text)
    if count == 0:
        _patch_failed_this_iter = True
        snippet = original[:300].replace("\n", "\\n")
        return (
            f"ERROR: old_text not found in {target}.\n"
            f"First 300 chars of the file: {snippet}"
        )
    if count > 1:
        # Find line numbers of all occurrences so the LLM can provide more context
        lines = original.splitlines()
        hits = [
            f"  line {i+1}: {lines[i].strip()}"
            for i, line in enumerate(lines)
            if old_text in line
        ]
        _patch_failed_this_iter = True
        return (
            f"ERROR: old_text appears {count} times in {target} — cannot patch "
            f"safely (would modify the wrong occurrence).\n"
            f"Occurrences:\n" + "\n".join(hits[:10]) + "\n\n"
            f"Supply a longer old_text that uniquely identifies the target line."
        )

    backup = target.with_suffix(".py.bak")
    backup.write_text(original)
    logging.info(f"BACKUP saved to {backup}")

    patched = original.replace(old_text, new_text, 1)
    target.write_text(patched)
    logging.info(
        f"PATCHED own emergence.py: replaced {len(old_text)} chars "
        f"with {len(new_text)} chars"
    )

    # Auto-verify: confirm new_text is actually present in the written file.
    # This gives the model concrete proof in the tool result itself, so it
    # doesn't have to run a separate grep_file to know the patch landed.
    verified_text = target.read_text()
    if new_text in verified_text:
        # Find the line number for context
        for lineno, line in enumerate(verified_text.splitlines(), 1):
            if new_text in line:
                verify_line = line.strip()
                verify_msg = f"\n✓ VERIFIED: new text found at line {lineno}: {verify_line!r}"
                break
        else:
            verify_msg = "\n✓ VERIFIED: new text is present in the file."
    else:
        _patch_failed_this_iter = True
        verify_msg = (
            "\n✗ VERIFICATION FAILED: new text NOT found after writing — "
            "the file may be corrupted. Do NOT proceed. Report this error to your peer."
        )

    # Write a restart goal so the new process knows what it was doing and what
    # to verify when it comes back online.  Consumed once at startup.
    try:
        goal_file = WORKING_DIR / "state" / "restart_goal.json"
        peer_num = 3 - INSTANCE_NUM

        # Detect role: canary = patches first; verifier = patches second.
        # If peer already wrote a restart_goal for this new_text, they went first → I'm the verifier.
        is_canary = True
        if PEER_DIR is not None:
            peer_goal_file = PEER_DIR / "state" / "restart_goal.json"
            if peer_goal_file.exists():
                try:
                    peer_data = json.loads(peer_goal_file.read_text())
                    if peer_data.get("new_text") == new_text:
                        is_canary = False  # peer already patched this change
                except Exception:
                    pass

        if is_canary:
            next_steps = [
                f"Call grep_file path=emergence.py pattern={new_text.split()[0]!r} "
                f"to confirm the patch is present in YOUR file.",
                f"Tell Instance {peer_num} you've restarted and the patch is confirmed.",
                f"Then call authorize_patch to grant Instance {peer_num} permission "
                f"to apply the same patch.",
                f"Tell Instance {peer_num} to apply the patch now.",
                f"After Instance {peer_num} restarts and confirms, YOU propose the "
                f"next improvement cycle as PROPOSER.",
            ]
        else:
            next_steps = [
                f"Call grep_file path=emergence.py pattern={new_text.split()[0]!r} "
                f"to confirm the patch is present in YOUR file.",
                f"Send Instance {peer_num}: 'Cycle complete — both instances updated. "
                f"You are the proposer for the next cycle — begin when ready.'",
                f"Wait for Instance {peer_num}'s proposal. Give genuine feedback.",
                f"Do NOT re-propose anything already patched "
                f"('{old_text.strip()}' was just changed).",
            ]
        goal_data = {
            "old_text": old_text,
            "new_text": new_text,
            "patched_at": datetime.now(timezone.utc).isoformat(),
            "instance_num": INSTANCE_NUM,
            "is_canary": is_canary,
            "next_steps": next_steps,
        }
        goal_file.write_text(json.dumps(goal_data, indent=2))
        logging.info(f"Restart goal written to {goal_file}")
    except Exception as ge:
        logging.warning(f"Could not write restart_goal.json: {ge}")

    return (
        f"OK: Patched {target}\n"
        f"  Replaced {len(old_text)} chars → {len(new_text)} chars\n"
        f"  File is now {len(patched)} bytes.\n"
        f"  Backup saved to {backup}"
        f"{verify_msg}\n\n"
        f"Now call restart_self to run updated code.\n"
        f"Tell your peer you are restarting so they can ping to confirm you came back."
    )


def tool_restore_own_file() -> str:
    """Restore THIS instance's own emergence.py from the pre-patch backup (.bak).

    Use this if your patch caused problems and you need to roll back.
    After restoring, call restart_self to relaunch with the known-good code.
    """
    target = WORKING_DIR / "emergence.py"
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        return f"ERROR: No backup found at {backup}. Cannot restore."

    original = backup.read_text()
    target.write_text(original)
    logging.info(f"RESTORED own emergence.py from backup")
    return (
        f"OK: Restored {target} from backup.\n"
        f"  File is now {len(original)} bytes (pre-patch version).\n\n"
        f"Call restart_self to relaunch with the known-good code."
    )


# Protected constants — patching these breaks the system's identity/networking
_PROTECTED_PATTERNS = [
    "INSTANCE_NUM =",
    "LISTEN_PORT =",
    "PEER_PORT =",
]


def tool_propose_patch(old_text: str, new_text: str, rationale: str) -> str:
    """Formally propose a code change to your peer for YES/NO approval.

    Python will:
      1. Validate the patch locally (unique match, no protected constants, syntax check)
      2. Send the proposal to your peer — they will approve or reject it
      3. If APPROVED: Python patches both files and restarts both instances automatically
      4. If REJECTED: returns peer's explanation so you can revise and re-propose

    You do NOT need to call patch_own_file, authorize_patch, or restart_self.
    After approval, Python handles all of that. You will restart automatically.
    """
    if PEER_DIR is None:
        return "ERROR: Peer not spawned yet. Call spawn_local_instance first."

    # Strip >>> display prefix if accidentally included
    clean_old = old_text
    if clean_old.lstrip().startswith(">>>"):
        clean_old = "\n".join(
            l[4:] if l.lstrip().startswith(">>> ") else l
            for l in clean_old.splitlines()
        ).strip()

    # Protected constants check
    for pat in _PROTECTED_PATTERNS:
        if pat in clean_old:
            return (
                f"ERROR: '{pat}' is a system-identity constant that must not be changed. "
                f"It controls instance numbering and network ports — patching it would "
                f"corrupt this instance's identity. Choose a different improvement."
            )

    # Local validation: unique match
    target = WORKING_DIR / "emergence.py"
    content = target.read_text()
    count = content.count(clean_old)
    if count == 0:
        snippet = content[:200].replace("\n", "\\n")
        return (
            f"ERROR: old_text not found in emergence.py.\n"
            f"Verify the exact text with grep_file first. "
            f"File starts with: {snippet}"
        )
    if count > 1:
        lines = content.splitlines()
        hits = [f"  line {i+1}: {lines[i].strip()}" for i, l in enumerate(lines) if clean_old in lines[i]]
        return (
            f"ERROR: old_text appears {count} times — too ambiguous.\n"
            f"Occurrences:\n" + "\n".join(hits[:8]) + "\n\nProvide more surrounding context."
        )

    # Syntax validation
    patched_content = content.replace(clean_old, new_text, 1)
    try:
        compile(patched_content, "emergence.py", "exec")
    except SyntaxError as e:
        return f"ERROR: Proposed patch creates a Python syntax error at line {e.lineno}: {e.msg}"

    # Write proposal to peer's state dir
    proposal = {
        "old_text": clean_old,
        "new_text": new_text,
        "rationale": rationale,
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "from_instance": INSTANCE_NUM,
        "proposer_working_dir": str(WORKING_DIR),
    }
    proposal_file = PEER_DIR / "state" / "patch_proposal.json"
    response_file = PEER_DIR / "state" / "patch_review_response.json"
    if response_file.exists():
        response_file.unlink()
    proposal_file.write_text(json.dumps(proposal, indent=2))
    logging.info(f"PROPOSE: Wrote patch proposal to {proposal_file}")

    # Poll for peer's response (3-minute timeout)
    deadline = time.time() + 180
    while time.time() < deadline:
        if response_file.exists():
            try:
                resp = json.loads(response_file.read_text())
                response_file.unlink()
                if proposal_file.exists():
                    proposal_file.unlink()
                if resp.get("approved"):
                    return _execute_approved_patch(clean_old, new_text, patched_content)
                else:
                    explanation = resp.get("explanation", "No reason given.")
                    return (
                        f"REJECTED by peer: {explanation}\n\n"
                        f"Revise your proposal and call propose_patch again, or use "
                        f"send_message to discuss the rejection with your peer."
                    )
            except Exception as e:
                logging.warning(f"PROPOSE: Error reading response: {e}")
        time.sleep(3)

    # Timeout
    if proposal_file.exists():
        proposal_file.unlink()
    return (
        "ERROR: Peer did not respond to patch proposal within 3 minutes. "
        "Use ping_peer to check if peer is alive."
    )


def _execute_approved_patch(old_text: str, new_text: str, patched_content: str) -> str:
    """Apply approved patch to both instances and trigger restarts. Internal only."""
    # Patch peer's file directly (peer = canary, patches first)
    peer_target = PEER_DIR / "emergence.py"
    peer_original = peer_target.read_text()
    if peer_original.count(old_text) != 1:
        return (
            f"ERROR: Peer's file has {peer_original.count(old_text)} occurrences of "
            f"old_text (expected 1). Cannot patch peer safely. Aborting."
        )
    peer_backup = peer_target.with_suffix(".py.bak")
    peer_backup.write_text(peer_original)
    peer_target.write_text(peer_original.replace(old_text, new_text, 1))
    logging.info("EXECUTE: Applied patch to peer's emergence.py")

    # Write restart_goal for peer (canary — proposes next cycle)
    peer_goal = {
        "old_text": old_text,
        "new_text": new_text,
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "instance_num": 3 - INSTANCE_NUM,
        "is_canary": True,
        "next_steps": [
            "Verify patch: grep_file path=emergence.py to confirm the change is present.",
            "Cycle complete — both instances are being updated and restarted.",
            "YOU are the PROPOSER for the next cycle. Begin when ready.",
        ],
    }
    (PEER_DIR / "state" / "restart_goal.json").write_text(json.dumps(peer_goal, indent=2))

    # Signal peer to restart
    (PEER_DIR / "state" / "restart_requested.flag").write_text(
        datetime.now(timezone.utc).isoformat()
    )
    logging.info("EXECUTE: Wrote restart_requested.flag to peer's state dir")

    # Patch own file
    target = WORKING_DIR / "emergence.py"
    (target.with_suffix(".py.bak")).write_text(target.read_text())
    target.write_text(patched_content)
    logging.info("EXECUTE: Applied patch to own emergence.py")

    # Write restart_goal for self (verifier — waits for peer's next proposal)
    own_goal = {
        "old_text": old_text,
        "new_text": new_text,
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "instance_num": INSTANCE_NUM,
        "is_canary": False,
        "next_steps": [
            "Verify patch: grep_file path=emergence.py to confirm the change is present.",
            "Cycle complete — both instances updated.",
            "Wait for your peer's next proposal (they are the proposer this cycle).",
        ],
    }
    (STATE_DIR / "restart_goal.json").write_text(json.dumps(own_goal, indent=2))

    # Restart self — this replaces the process; no code runs after this
    logging.info("EXECUTE: Restarting self after successful patch.")
    tool_restart_self()
    return "Restarting..."  # unreachable


def tool_review_proposal(approved: bool, explanation: str) -> str:
    """Respond to a patch proposal from your peer.

    approved   : true to approve, false to reject
    explanation: brief reason (shown to peer if rejected; logged if approved)

    After approving: you will restart automatically shortly.
    You do NOT need to call any other tools after this.
    """
    # Clean up proposal file
    proposal_file = STATE_DIR / "patch_proposal.json"
    if proposal_file.exists():
        proposal_file.unlink()

    # Write response for proposer to pick up
    response_file = STATE_DIR / "patch_review_response.json"
    response_file.write_text(json.dumps({
        "approved": approved,
        "explanation": explanation,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }))
    logging.info(f"REVIEW: {'Approved' if approved else 'Rejected'} — {explanation}")

    if approved:
        return (
            "OK: Patch approved. The proposer is applying patches to both instances "
            "and will restart both. You will restart shortly — no further action needed."
        )
    else:
        return f"OK: Rejection recorded with explanation: '{explanation}'"


def tool_check_resources() -> str:
    """Check system resources: available RAM and running emergence instance count.

    Each model instance needs ~6 GB RAM. Two instances is the recommended maximum.
    Check before any operation that would start a new process.
    """
    lines = ["System resource check:"]
    free_gb = 0

    try:
        vm = subprocess.run(
            "vm_stat", shell=True, capture_output=True, text=True, timeout=5
        )
        free_pages = 0
        for line in vm.stdout.splitlines():
            if "Pages free" in line or "Pages inactive" in line:
                val = line.split(":")[-1].strip().rstrip(".")
                try:
                    free_pages += int(val)
                except ValueError:
                    pass
        free_gb = (free_pages * 4096) / (1024 ** 3)
        lines.append(f"  Free RAM (approx): {free_gb:.1f} GB")
    except Exception as e:
        lines.append(f"  Free RAM: unknown ({e})")

    try:
        ps = subprocess.run(
            "pgrep -f 'python.*emergence\\.py' | wc -l",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        count = int(ps.stdout.strip())
        lines.append(f"  Emergence instances running: {count}")
    except Exception:
        count = -1
        lines.append("  Emergence instances running: unknown")

    if free_gb > 8 and (count < 0 or count < 3):
        advice = "Resources OK. A restart or third instance could be supported."
    elif free_gb > 4 or count <= 2:
        advice = "2 instances is the safe maximum. Do NOT spawn additional processes."
    else:
        advice = "⚠ Low resources. Do NOT spawn any additional instances."

    lines.append(f"  Advice: {advice}")
    return "\n".join(lines)


def _silent_ping() -> bool:
    """Ping the peer instance. Returns True if alive, False otherwise. No side effects."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PEER_PORT}/ping", timeout=3
        ) as resp:
            json.loads(resp.read())
            return True
    except Exception:
        return False


def tool_ping_peer() -> str:
    """Check if the peer instance is alive. Returns its status JSON or an error."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PEER_PORT}/ping", timeout=5
        ) as resp:
            data = json.loads(resp.read())
            peer_num = 3 - INSTANCE_NUM
            logging.info(f"PING: Instance {peer_num} is ALIVE — {data}")
            return (
                f"Instance {peer_num} is ALIVE.\n"
                f"  ID          : {data.get('instance_id')}\n"
                f"  Working dir : {data.get('working_dir')}\n"
                f"  Port        : {data.get('port', PEER_PORT)}"
            )
    except Exception as e:
        peer_num = 3 - INSTANCE_NUM
        logging.warning(f"PING: Instance {peer_num} did not respond — {e}")
        return (
            f"Instance {peer_num} is NOT responding on port {PEER_PORT}.\n"
            f"Reason: {e}\n"
            f"They may still be loading the model (can take 30-60s). "
            f"Continue your own work and try again later."
        )


def _start_heartbeat() -> None:
    """Background thread: ping the peer every 30 s and log status to state/."""
    import urllib.request

    def _loop():
        peer_num = 3 - INSTANCE_NUM
        status_file = STATE_DIR / "peer_status.json"
        while True:
            time.sleep(30)
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PEER_PORT}/ping", timeout=5
                ) as resp:
                    data = json.loads(resp.read())
                    record = {
                        "alive": True,
                        "checked": datetime.now(timezone.utc).isoformat(),
                        "instance_id": data.get("instance_id"),
                    }
                    global _peer_last_seen_alive
                    _peer_last_seen_alive = time.time()
                    logging.info(
                        f"HEARTBEAT: Instance {peer_num} ALIVE "
                        f"(id={data.get('instance_id')})"
                    )
            except Exception as e:
                record = {
                    "alive": False,
                    "checked": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                }
                logging.warning(f"HEARTBEAT: Instance {peer_num} NOT responding — {e}")
            try:
                status_file.write_text(json.dumps(record, indent=2))
            except Exception:
                pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def _build_tools() -> dict:
    """Build the tool registry. Instance-specific tools are added conditionally."""
    peer_host = "127.0.0.1"
    tools = {
        "shell": lambda args: tool_shell(args.get("command", "")),
        "read_file": lambda args: tool_read_file(args.get("path", "")),
        "write_file": lambda args: tool_write_file(
            args.get("path", ""), args.get("content", "")
        ),
        "list_dir": lambda args: tool_list_dir(args.get("path", ".")),
        "send_message": lambda args: tool_send_message(
            args.get("host", peer_host),
            args.get("port", PEER_PORT),
            # Accept "message" or "content" — the model uses both
            args.get("message") or args.get("content", ""),
        ),
        "grep_file": lambda args: tool_grep_file(
            args.get("path", ""), args.get("pattern", ""), args.get("context", 2)
        ),
        "scan_network": lambda args: tool_scan_network(args.get("subnet", "")),
        "ping_peer": lambda args: tool_ping_peer(),
    }
    # Self-patching tools — each instance patches and restores its own file
    tools["patch_own_file"] = lambda args: tool_patch_own_file(
        args.get("old_text", ""), args.get("new_text", "")
    )
    tools["restore_own_file"] = lambda args: tool_restore_own_file()
    tools["restart_self"] = lambda args: tool_restart_self()
    tools["check_resources"] = lambda args: tool_check_resources()
    if INSTANCE_NUM == 1:
        tools["spawn_local_instance"] = lambda args: tool_spawn_local_instance()
    tools["authorize_patch"] = lambda args: tool_authorize_patch()
    # Both instances can verify and recover their peer's file
    tools["restore_peer_file"] = lambda args: tool_restore_peer_file()
    tools["grep_peer_file"] = lambda args: tool_grep_peer_file(
        args.get("pattern", ""), args.get("context", 1)
    )
    tools["propose_patch"] = lambda args: tool_propose_patch(
        args.get("old_text", ""), args.get("new_text", ""), args.get("rationale", "")
    )
    tools["review_proposal"] = lambda args: tool_review_proposal(
        args.get("approved", False), args.get("explanation", "")
    )
    tools["read_file_lines"] = lambda args: tool_read_file_lines(
        args.get("path", ""), args.get("start", 1), args.get("end", 80)
    )
    tools["file_outline"] = lambda args: tool_file_outline(args.get("path", ""))
    return tools

# ---------------------------------------------------------------------------
# Communication Listener — receives messages from sibling instances
# ---------------------------------------------------------------------------

class MessageHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler for receiving messages from other instances."""

    def do_POST(self):
        if self.path == "/message":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                msg = json.loads(body)
                from_inst = msg.get("from_instance", "?")
                content = msg.get("message", "")
                logging.info(f"RECEIVED from Instance {from_inst}: {content[:2000]}")
                # Note: log_comm is NOT called here — it's already called in
                # tool_send_message on the sending side, so logging it here too
                # would cause every message to appear twice in the comm log.
                # Receiving a message proves the sender is alive
                global _peer_last_seen_alive
                _peer_last_seen_alive = time.time()
                # Save to inbox
                inbox_dir = STATE_DIR / "inbox"
                inbox_dir.mkdir(exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                (inbox_dir / f"msg_{ts}_{msg.get('from', 'unknown')}.json").write_text(
                    json.dumps(msg, indent=2)
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "received"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f'{{"error": "{e}"}}'.encode())
        elif self.path == "/ping":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "instance": INSTANCE_NUM,
                "instance_id": INSTANCE_ID,
                "host": socket.gethostname(),
                "alive": True,
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/ping":
            # Receiving a ping request means the sender is alive — record it.
            # This gives I1 awareness of I2 even when I2's HTTP listener is
            # temporarily down (e.g. during the old→new process handoff).
            global _peer_last_seen_alive
            _peer_last_seen_alive = time.time()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "instance": INSTANCE_NUM,
                "instance_id": INSTANCE_ID,
                "host": socket.gethostname(),
                "working_dir": str(WORKING_DIR),
                "alive": True,
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging


def start_listener():
    """Start the HTTP listener in a background thread.

    Retries up to 5 times with 3s gaps to handle the race where an old
    process is still holding the port as it exits (restart_self exits after 3s).
    """
    for attempt in range(6):
        try:
            server = http.server.HTTPServer(("0.0.0.0", LISTEN_PORT), MessageHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            logging.info(f"Listening on port {LISTEN_PORT}")
            return server
        except OSError as e:
            if attempt < 5:
                logging.info(
                    f"Port {LISTEN_PORT} in use — retrying in 3s "
                    f"({attempt + 1}/6)…"
                )
                time.sleep(3)
            else:
                logging.warning(
                    f"Could not start listener on port {LISTEN_PORT}: {e}"
                )
                return None

# ---------------------------------------------------------------------------
# LLM Interface — loads and queries the local model via MLX
# ---------------------------------------------------------------------------

class LocalLLM:
    """Wrapper around mlx-lm for local inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model. Downloads if not cached."""
        logging.info(f"Loading model: {self.model_name}")
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_name)
            logging.info("Model loaded successfully.")
        except ImportError:
            logging.error(
                "mlx-lm is not installed. Run: pip install mlx-lm\n"
                "Then re-run this script."
            )
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            sys.exit(1)

    def generate(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Generate a response given a conversation history.

        Applies prompt repetition (Promptless Reasoning, 2025): the full formatted
        prompt body is repeated before the generation token.  For causal LMs this
        lets tokens near the end attend to tokens at the start via the repeated
        copy, improving instruction-following without increasing generated-token
        count or latency.  Effect is positive for non-reasoning models and neutral
        for reasoning models, so it's always safe to apply.
        """
        from mlx_lm import generate

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Build prompt body (no generation prompt) and generation suffix separately
            # so we can repeat only the body: body + body + gen_suffix.
            try:
                prompt_body = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_with_gen = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                gen_suffix = prompt_with_gen[len(prompt_body):]
                prompt = prompt_body + prompt_body + gen_suffix
            except Exception:
                # Fallback: some tokenizers don't support add_generation_prompt=False
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|{role}|>\n{content}")
            parts.append("<|assistant|>\n")
            body = "\n".join(parts[:-1])
            prompt = body + "\n" + body + "\n" + parts[-1]

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response

# ---------------------------------------------------------------------------
# Agentic Loop — the core cycle of reason → act → observe → repeat
# ---------------------------------------------------------------------------

def parse_all_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract ALL tool calls from LLM output. Returns list of (tool_name, args) pairs.

    Two passes:
    1. Fenced code blocks  (```tool / ```json / ```python)
    2. Bare JSON objects   (model sometimes omits the fences entirely)

    Both styles are handled so Instance 2 (which outputs raw JSON) works the
    same as Instance 1 (which wraps JSON in backtick fences).

    Pre-processing: strip any hallucinated "[MESSAGE FROM INSTANCE N]:" or
    "⟦PEER MSG⟧" blocks the model may generate in its own text.  These are the
    inbox injection markers — if the model reproduces them it creates phantom
    tool calls that get executed spuriously.
    """
    import re

    # Strip hallucinated injection-format blocks before the bare-JSON pass.
    # Pattern covers both the old "[MESSAGE FROM INSTANCE N]:" format and the
    # new "⟦PEER MSG⟧" format.  We remove from the marker up to (but not
    # including) the next double newline or end of string, so only the phantom
    # JSON inside gets discarded, not surrounding real content.
    _hallucination_pattern = re.compile(
        r"(?:\[(?:MESSAGE FROM|MSG #\d+ FROM) INSTANCE \d+[^\]]*\]"
        r"|⟦PEER MSG[^⟧]*⟧)"
        r"[^\n]*",   # rest of the same line after the marker
        re.IGNORECASE,
    )
    text_for_bare_json = _hallucination_pattern.sub("", text)

    results: list[tuple[str, dict]] = []
    matched_spans: list[tuple[int, int]] = []  # track already-consumed ranges

    # ── Pass 1: fenced blocks (use original text — fences are unambiguous) ───
    fence_pattern = r"```(?:tool|json|python)?\s*\n?\s*(\{.*?\})\s*\n?\s*```"
    for match in re.finditer(fence_pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "tool" in call:
                results.append((call.get("tool"), call.get("args", {})))
                matched_spans.append((match.start(), match.end()))
        except json.JSONDecodeError:
            continue

    # ── Pass 2: bare JSON objects (on hallucination-stripped text) ───────────
    # Use raw_decode so we handle nested braces correctly without a full
    # regex. We try to parse at every `{` that isn't inside a span we've
    # already consumed.  Using text_for_bare_json here ensures we don't pick
    # up tool calls that only appeared inside hallucinated injection blocks.
    decoder = json.JSONDecoder()
    for brace in re.finditer(r'\{', text_for_bare_json):
        pos = brace.start()
        # Skip anything already covered by a fenced block or prior bare match
        if any(s <= pos < e for s, e in matched_spans):
            continue
        try:
            obj, end_idx = decoder.raw_decode(text_for_bare_json, pos)
            if isinstance(obj, dict) and "tool" in obj:
                results.append((obj.get("tool"), obj.get("args", {})))
                # Mark this span consumed so inner braces aren't re-parsed
                matched_spans.append((pos, end_idx))
        except (json.JSONDecodeError, ValueError):
            continue

    return results


def run_agentic_loop(llm: LocalLLM, max_iterations: int = MAX_ITERATIONS):
    """The main loop. The LLM reasons, optionally calls tools, observes results."""
    tools = _build_tools()

    messages = [
        {"role": "system", "content": get_system_prompt()},
    ]

    logging.info("=" * 60)
    logging.info(f"EMERGENCE — Instance {INSTANCE_NUM} starting")
    logging.info(f"Instance ID : {INSTANCE_ID}")
    logging.info(f"Host        : {socket.gethostname()}")
    logging.info(f"Model       : {llm.model_name}")
    logging.info(f"My port     : {LISTEN_PORT}")
    logging.info(f"Peer port   : {PEER_PORT}")
    logging.info(f"Working dir : {WORKING_DIR}")
    logging.info(f"Logs also in: /tmp/emergence_instance_{INSTANCE_NUM}_{INSTANCE_ID}.log")
    logging.info(f"Comm log    : {COMM_LOG}")
    logging.info("=" * 60)

    no_action_streak = 0
    chatter_streak = 0

    # Clear any stale inbox messages left over from a previous run.
    # Without this, I1 starts up seeing phantom messages from a previous I2
    # and skips the spawn step entirely.
    inbox_dir = STATE_DIR / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for stale in inbox_dir.glob("*.json"):
        try:
            stale.unlink()
        except Exception:
            pass
    logging.info("Inbox cleared (fresh start).")

    # ── Restart goal injection ────────────────────────────────────────────────
    # If this instance just restarted after applying a patch, patch_own_file
    # wrote a goal file so we know exactly what was done and what to verify.
    # tool_restart_self also saved a conversation tail so context is preserved.
    # Consume it once here so the LLM starts with the right context.
    goal_file = STATE_DIR / "restart_goal.json"
    restart_goal_loaded = False
    goal_data = {}
    if goal_file.exists():
        try:
            goal_data = json.loads(goal_file.read_text())
            goal_file.unlink()  # one-time use — don't inject on future iterations
            steps = "\n".join(
                f"  {i+1}. {s}"
                for i, s in enumerate(goal_data.get("next_steps", []))
            )
            # Include the saved conversation tail so the instance remembers
            # what was being discussed before the restart.
            tail = goal_data.get("conversation_tail", [])
            tail_text = ""
            if tail:
                tail_lines = []
                for m in tail[-6:]:  # show last 6 messages
                    role = m.get("role", "?").upper()
                    content = m.get("content", "")[:400]
                    tail_lines.append(f"[{role}]: {content}")
                tail_text = (
                    "\n\nConversation context from before restart "
                    "(most recent last):\n"
                    + "\n\n".join(tail_lines)
                )
            messages.append({
                "role": "user",
                "content": (
                    "📋 RESTART CONTEXT — You just restarted after applying a patch.\n\n"
                    f"Change applied:\n"
                    f"  old: {goal_data.get('old_text', '?')!r}\n"
                    f"  new: {goal_data.get('new_text', '?')!r}\n"
                    f"Patched at: {goal_data.get('patched_at', '?')}\n\n"
                    f"Your immediate tasks:\n{steps}"
                    f"{tail_text}\n\n"
                    "Do these NOW before anything else."
                ),
            })
            restart_goal_loaded = True
            logging.info("Restart goal injected from state/restart_goal.json")
        except Exception as e:
            logging.warning(f"Could not read restart_goal.json: {e}")

    # ── I1 only: inject real constants snapshot ───────────────────────────────
    # I1 tends to propose changes without reading the file, and the system
    # prompt's example old_text values bias it toward specific values.
    # Grepping the real constants at startup gives I1 actual data to work from.
    # Context-aware: if we restarted after a patch, peer is already running —
    # do NOT tell I1 to call spawn_local_instance.
    if INSTANCE_NUM == 1:
        try:
            snapshot = tool_grep_file("emergence.py", r"^[A-Z][A-Z_]+ = ", context=0)
            if restart_goal_loaded:
                patched_old = goal_data.get("old_text", "")
                patched_new = goal_data.get("new_text", "")
                followup = (
                    f"Your peer (Instance 2) is already running on port {PEER_PORT}. "
                    "Do NOT call spawn_local_instance. Follow your restart tasks above.\n\n"
                    f"⚠ ALREADY PATCHED THIS SESSION: {patched_old!r} → {patched_new!r}\n"
                    "Do NOT re-propose this or any other already-changed value. "
                    "Your next improvement must target something genuinely DIFFERENT."
                )
            else:
                followup = (
                    "Now follow your role instructions: call spawn_local_instance first.\n\n"
                    "⚠ Do NOT propose changing MAX_ITERATIONS or MAX_TOOL_OUTPUT if they "
                    "already show non-default values above. Pick something genuinely different."
                )
            messages.append({
                "role": "user",
                "content": (
                    "Current constant values in YOUR emergence.py "
                    "(auto-read at startup — base proposals on these, "
                    "not on the system prompt examples):\n\n"
                    f"{snapshot}\n\n"
                    f"{followup}"
                ),
            })
        except Exception:
            pass

    # ── I2 only: enforce first-response discipline ────────────────────────────
    # I2 often runs grep_file in iteration 1 but generates its analysis as plain
    # text with no send_message call — so I1 never hears I2's independent view.
    # This injection makes the requirement explicit immediately before generation.
    # SKIP if restart_goal was loaded — the goal already tells I2 what to do,
    # and overriding it with FIRST RESPONSE REQUIREMENT causes I2 to re-negotiate
    # instead of confirming the patch.
    if INSTANCE_NUM == 2:
        if restart_goal_loaded:
            messages.append({
                "role": "user",
                "content": (
                    "⚠ IMMEDIATE ACTION — do this in your FIRST response:\n"
                    "  1. grep_file path=emergence.py to confirm the patch is present\n"
                    "  2. send_message to tell Instance 1 you are online and the patch "
                    "is confirmed\n"
                    "Both as ```tool blocks in the same response."
                ),
            })
        else:
            inbox_has_messages = bool(list(inbox_dir.glob("*.json")))
            if inbox_has_messages:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠ FIRST RESPONSE REQUIREMENT:\n"
                        "Your very first response MUST include BOTH of these tool calls:\n"
                        "  1. grep_file with path=emergence.py and pattern='^[A-Z][A-Z_]+ = ' context=0\n"
                        "  2. send_message with your analysis of what you found and what YOU suggest\n\n"
                        "Both must appear as ```tool blocks in the same response."
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠ INBOX EMPTY — Your peer has not yet contacted you.\n\n"
                        "Do NOT call grep_file, patch_own_file, or send_message yet.\n"
                        "Your ONLY action this iteration is to wait. No tool call is needed.\n"
                        "When your peer contacts you, a ⟦PEER MSG⟧ block will appear "
                        "in your next iteration — reply to that."
                    ),
                })

    for iteration in range(max_iterations):
        logging.info(f"--- Iteration {iteration + 1}/{max_iterations} ---")

        # ── STEP 0 (I1 only): Python-level peer enforcement ──────────────────
        # Runs BEFORE the LLM every iteration.  Decisions about spawning or
        # waiting are made here in Python so the LLM can't get confused.
        if INSTANCE_NUM == 1:
            global _peer_restart_pending
            peer_alive = _silent_ping()

            if peer_alive:
                # ── Peer is UP ────────────────────────────────────────────
                if _peer_restart_pending:
                    # Peer just came back from a restart — tell the LLM
                    _peer_restart_pending = False
                    logging.info("Peer is back online after restart.")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"✓ Your peer is back online on port {PEER_PORT} "
                            f"after restarting.\n\n"
                            "⚠ IMPORTANT: Your peer has FRESH CONTEXT — it does NOT "
                            "remember the negotiation from before the restart. "
                            "Its restart_goal will guide it automatically.\n\n"
                            "Your FIRST message to it should re-introduce the situation:\n"
                            "  'You just restarted with the updated code. Please verify "
                            "with grep_file that the change is present in your file, "
                            "then confirm you are running correctly.'\n\n"
                            "Check YOUR restart_goal: if it says to call authorize_patch "
                            "for your peer, do so now and tell them to apply the patch."
                        ),
                    })
            else:
                # ── Peer is DOWN ──────────────────────────────────────────
                now = time.time()

                if PEER_DIR is None:
                    # Never spawned at all.
                    for stale in inbox_dir.glob("*.json"):
                        try:
                            stale.unlink()
                        except Exception:
                            pass
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⚠ PEER NOT RUNNING — Instance 2 is not online on "
                            f"port {PEER_PORT}.\n"
                            "Your ONLY action this iteration must be "
                            "spawn_local_instance.\n"
                            "Do not read files, do not send messages, do not do "
                            "anything else first. Call spawn_local_instance NOW."
                        ),
                    })

                else:
                    # Peer was spawned.  Check for an explicit restart flag.
                    restart_flag = PEER_DIR / "state" / "restarting.flag"

                    if restart_flag.exists():
                        # I2 called restart_self — intentional, expected gap.
                        _peer_restart_pending = True
                        try:
                            flag_ts = datetime.fromisoformat(
                                restart_flag.read_text().strip()
                            ).timestamp()
                        except Exception:
                            flag_ts = now - 10
                        elapsed = now - flag_ts
                        remaining = max(0, 120 - elapsed)

                        if elapsed < 120:
                            logging.info(
                                f"I2 restart in progress — {elapsed:.0f}s elapsed, "
                                f"~{remaining:.0f}s patience remaining. "
                                "Polling every 5s."
                            )
                            # Skip LLM only if inbox is empty — if messages
                            # arrived (e.g. from I2's new process) run the LLM.
                            if not list(inbox_dir.glob("*.json")):
                                time.sleep(5)
                                continue   # fast-poll, skip LLM this iteration

                        else:
                            # Restart has taken too long — something went wrong.
                            logging.warning(
                                f"I2 restart flag is {elapsed:.0f}s old — "
                                "timeout exceeded. Triggering recovery."
                            )
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"⚠ RESTART TIMEOUT — Instance 2 signalled a "
                                    f"restart {elapsed:.0f}s ago but port {PEER_PORT} "
                                    f"is still not responding.\n"
                                    "Steps to recover:\n"
                                    "1. Call restore_peer_file to revert to the "
                                    "known-good code.\n"
                                    "2. Call spawn_local_instance to start a fresh "
                                    "Instance 2."
                                ),
                            })

                    else:
                        # No restart flag — use last-contact window.
                        last_contact = max(_peer_last_seen_alive, _peer_spawn_time)
                        gap = now - last_contact
                        if gap < 150:
                            logging.info(
                                f"Peer offline — {gap:.0f}s since last contact, "
                                "within patience window. Polling every 5s."
                            )
                            if not list(inbox_dir.glob("*.json")):
                                time.sleep(5)
                                continue
                        else:
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"⚠ PEER OFFLINE — Instance 2 has not responded "
                                    f"for {gap:.0f}s on port {PEER_PORT}.\n"
                                    "Call spawn_local_instance to start a fresh "
                                    "Instance 2."
                                ),
                            })

        # ── AUTH FLAG ENFORCEMENT (both instances) ───────────────────────────
        # If our peer has already written patch_authorized.flag to our state dir,
        # the LLM MUST call patch_own_file this iteration — no more discussion.
        auth_flag_path = STATE_DIR / "patch_authorized.flag"
        if auth_flag_path.exists():
            peer_num_ef = 3 - INSTANCE_NUM
            flag_age = time.time() - auth_flag_path.stat().st_mtime
            logging.info(
                f"AUTH FLAG present in our state dir (age {flag_age:.0f}s) — "
                "injecting patch execution reminder."
            )
            messages.append({
                "role": "user",
                "content": (
                    f"⚠ EXECUTE PATCH NOW — Instance {peer_num_ef} has called "
                    f"authorize_patch for you. The authorization flag is in your "
                    f"state directory (age {flag_age:.0f}s).\n\n"
                    f"Your ONLY action this iteration:\n"
                    f"  1. Call patch_own_file with the agreed old_text and new_text.\n"
                    f"  2. Call grep_file to verify the change.\n"
                    f"  3. Tell Instance {peer_num_ef} you are restarting.\n"
                    f"  4. Call restart_self.\n\n"
                    f"Do NOT send more messages first. Do NOT discuss further. "
                    f"Apply the patch now."
                ),
            })

        # ── PATCH PROPOSAL DETECTION (both instances) ────────────────────────────
        # If our peer wrote a patch_proposal.json to our state dir, inject a review prompt.
        proposal_file_path = STATE_DIR / "patch_proposal.json"
        if proposal_file_path.exists():
            try:
                prop = json.loads(proposal_file_path.read_text())
                peer_num_prop = prop.get("from_instance", "?")
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠ PATCH REVIEW REQUIRED — Instance {peer_num_prop} proposes:\n\n"
                        f"OLD TEXT:\n```\n{prop['old_text']}\n```\n\n"
                        f"NEW TEXT:\n```\n{prop['new_text']}\n```\n\n"
                        f"RATIONALE: {prop.get('rationale', '(none)')}\n\n"
                        f"Before deciding: verify old_text exists with "
                        f"grep_file path=emergence.py.\n\n"
                        f"Your ONLY action this iteration is to call review_proposal:\n"
                        f"  review_proposal approved=true explanation='...'  — to approve\n"
                        f"  review_proposal approved=false explanation='...'  — to reject\n\n"
                        f"Think carefully: Is this change safe? Does it improve the system?\n"
                        f"If approved, Python will patch and restart both instances automatically."
                    ),
                })
            except Exception as e:
                logging.warning(f"PROPOSAL: Could not read proposal file: {e}")

        # ── RESTART REQUEST DETECTION (both instances) ────────────────────────────
        # If the proposer patched our file and set this flag, restart immediately.
        restart_req_path = STATE_DIR / "restart_requested.flag"
        if restart_req_path.exists():
            restart_req_path.unlink()
            logging.info("RESTART: restart_requested.flag detected — restarting per proposer's instruction.")
            tool_restart_self()

        # ── STEP 1: Drain inbox BEFORE generating ────────────────────────────
        # Messages are always injected here, regardless of whether the last
        # iteration made a tool call. This ensures the LLM never misses a reply.
        if inbox_dir.exists():
            pending = sorted(inbox_dir.glob("*.json"))
            # Cap to 3 messages per iteration — injecting 10+ at once overwhelms
            # the context window and breaks coherence.  Extras stay in inbox
            # and will be injected in the next iteration.
            if len(pending) > 3:
                logging.info(
                    f"Inbox has {len(pending)} messages — injecting first 3, "
                    "deferring the rest to the next iteration."
                )
                pending = pending[:3]
            for msg_file in pending:
                try:
                    msg_data = json.loads(msg_file.read_text())
                    from_inst = msg_data.get("from_instance", "?")
                    text = msg_data.get("message", "")
                    from_port = msg_data.get("from_port", PEER_PORT)
                    # If the peer says they are restarting, note it so the
                    # enforcement loop can inject a context-handoff when they
                    # come back online.
                    if INSTANCE_NUM == 1 and any(
                        kw in text.lower()
                        for kw in ("restarting", "restart_self", "i'm restarting")
                    ):
                        _peer_restart_pending = True
                        logging.info(
                            "Detected restart announcement from peer — "
                            "will notify I1 when I2 is back online."
                        )
                    seq = msg_data.get("seq", "?")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⟦PEER MSG #{seq} | Instance {from_inst} | port {from_port}⟧\n"
                            f"{text}\n"
                            f"⟦END PEER MSG⟧\n\n"
                            f"Reply NOW with send_message (host=127.0.0.1 port={from_port}) "
                            "before any other action."
                        ),
                    })
                    msg_file.unlink()
                    no_action_streak = 0
                    logging.info(f"Injected msg #{seq} from Instance {from_inst} (port {from_port}) into conversation")
                except Exception:
                    pass

        # ── STEP 2: Generate ─────────────────────────────────────────────────
        try:
            response = llm.generate(messages, max_tokens=2048)
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            time.sleep(5)
            continue

        # Log the full response so reasoning is visible in the log files.
        # Long responses are split across multiple log lines (2000 chars each).
        for _chunk_start in range(0, len(response), 2000):
            _chunk = response[_chunk_start:_chunk_start + 2000]
            if _chunk_start == 0:
                logging.info(f"LLM: {_chunk}")
            else:
                logging.info(f"LLM (cont): {_chunk}")

        # ── Hallucination guard ───────────────────────────────────────────────
        # If the response STARTS with an injection marker the model is roleplaying
        # as the peer rather than acting as itself.  Discard the response and
        # inject a corrective warning.  This catches the most common failure mode
        # where the model begins its output with "[MESSAGE FROM INSTANCE N]:" or
        # the new "⟦PEER MSG⟧" format.
        _HALLU_PREFIXES = (
            "[message from instance",
            "[msg #",
            "⟦peer msg",
        )
        if response.strip().lower().startswith(_HALLU_PREFIXES):
            logging.warning(
                f"HALLUCINATION: LLM response started with injection marker — "
                f"checking for salvageable tool calls. Preview: {response[:120]!r}"
            )
            tool_block_idx = response.find("```tool")
            if tool_block_idx > 0:
                # Real tool calls exist after the hallucinated preamble.
                # Strip the preamble and fall through to normal processing.
                logging.info(
                    "HALLUCINATION: Stripping hallucinated preamble; "
                    "processing tool blocks from remainder."
                )
                response = response[tool_block_idx:]
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠ HALLUCINATION (auto-corrected) — Your previous response "
                        "began with a simulated peer message that was stripped. "
                        "Do NOT simulate peer messages. "
                        "Start your responses directly with your reasoning or a tool block."
                    ),
                })
                # Fall through — response now begins at the tool block.
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠ HALLUCINATION — Your previous response began with a peer "
                        "message marker. You cannot receive messages by generating them. "
                        "Peer messages only arrive as injected ⟦PEER MSG⟧ blocks before "
                        "your response.\n\n"
                        "Generate a real response with an actual tool call "
                        "(e.g. send_message, ping_peer, grep_file). "
                        "Do NOT simulate your peer's replies."
                    ),
                })
                no_action_streak += 1
                time.sleep(1)
                continue

        messages.append({"role": "assistant", "content": response})

        # Update the rolling conversation tail for restart context.
        # We keep non-system messages so the restarted instance sees the dialogue.
        global _conversation_tail
        _conversation_tail = [
            m for m in messages[-12:] if m.get("role") != "system"
        ]

        # ── STEP 3: Execute ALL tool calls found in the response ─────────────
        # The model sometimes emits several ```tool blocks in one reply
        # (e.g. patch_own_file then send_message). We execute every one
        # in order so nothing gets silently dropped.
        # Hard cap: max 5 calls per response. Prevents runaway loops where the
        # model emits dozens of ping/send calls in one generation.
        all_calls = parse_all_tool_calls(response)

        # Deduplicate consecutive identical tool calls (7B model sometimes emits the
        # same block twice in one response).
        deduped = []
        for call in all_calls:
            if not deduped or call != deduped[-1]:
                deduped.append(call)
        if len(deduped) < len(all_calls):
            logging.info(
                f"Deduped {len(all_calls) - len(deduped)} duplicate tool call(s)."
            )
        all_calls = deduped

        MAX_CALLS_PER_RESPONSE = 5
        if len(all_calls) > MAX_CALLS_PER_RESPONSE:
            logging.warning(
                f"Truncating {len(all_calls)} tool calls to {MAX_CALLS_PER_RESPONSE} "
                "per response cap."
            )
            all_calls = all_calls[:MAX_CALLS_PER_RESPONSE]

        if all_calls:
            all_results = []
            global _patch_failed_this_iter
            _patch_failed_this_iter = False  # Reset per-response flag before executing tools
            for tool_name, tool_args in all_calls:
                if tool_name in tools:
                    logging.info(
                        f"{_MAGENTA}TOOL:{_RESET} {tool_name}({json.dumps(tool_args)[:200]})"
                    )
                    result = tools[tool_name](tool_args)
                    logging.info(f"RESULT: {result[:300]}{'...' if len(result) > 300 else ''}")
                    all_results.append(f"Tool result for {tool_name}:\n{result}")
                    # Once restart_self fires, this process is exiting in ~3s.
                    # Stop executing further tools — they would be meaningless
                    # and can cause avalanche effects (ping loops, double restarts).
                    if tool_name == "restart_self" and _restart_initiated:
                        if len(all_calls) > all_calls.index((tool_name, tool_args)) + 1:
                            skipped = len(all_calls) - all_calls.index((tool_name, tool_args)) - 1
                            logging.info(
                                f"Halting after restart_self — skipping {skipped} "
                                "remaining tool call(s)."
                            )
                        break
                else:
                    logging.warning(f"Unknown tool requested: {tool_name}")
                    all_results.append(
                        f"Unknown tool: '{tool_name}'. "
                        f"Available tools: {', '.join(tools.keys())}\n"
                        "Note: there is no receive_message tool. Messages from your "
                        "peer arrive automatically at the start of each iteration."
                    )
            combined = "\n\n---\n\n".join(all_results)

            # Detect chatter: all tool calls were send_message with no action tool
            _CHATTER_TOOLS = {"send_message", "ping_peer", "check_resources"}
            only_chatter = all(name in _CHATTER_TOOLS for name, _ in all_calls)
            if only_chatter:
                chatter_streak += 1
            else:
                chatter_streak = 0

            # ── False-restart detection ───────────────────────────────────────
            # If any send_message in this response claimed "restarting" but there
            # was no restart_self call in the same response, the model is lying.
            # This is the critical failure mode where I2 says "I'm restarting now"
            # without ever calling patch_own_file or restart_self.
            tool_names_this_response = {name for name, _ in all_calls}
            restart_claimed = any(
                name == "send_message"
                and any(
                    kw in args.get("message", "").lower()
                    for kw in ("restarting now", "i'm restarting", "restart_self", "will restart")
                )
                for name, args in all_calls
            )
            restart_executed = "restart_self" in tool_names_this_response
            if restart_claimed and not restart_executed:
                logging.warning(
                    "FALSE RESTART DETECTED: response claimed restart but no restart_self call."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠ FALSE RESTART — Your message just said you are restarting, "
                        "but your response did NOT contain a restart_self tool call. "
                        "Saying you will restart is NOT restarting.\n\n"
                        "Look at the grep_file results above. If the agreed new value is "
                        "NOT shown in your file, you have NOT called patch_own_file yet.\n\n"
                        "Your IMMEDIATE next response must contain ALL of these in order:\n"
                        "  1. patch_own_file with the agreed old_text and new_text\n"
                        "  2. grep_file to verify the change is in your file\n"
                        "  3. send_message telling Instance 1 the result\n"
                        "  4. restart_self\n\n"
                        "Do NOT send a message claiming the patch is done until "
                        "grep_file shows the new value. The order matters."
                    ),
                })

            messages.append({
                "role": "user",
                "content": combined + "\n\nWhat do you observe? What is your next action?",
            })
            no_action_streak = 0

            if chatter_streak >= 6:
                # Hard escalation — the loop is completely stuck
                messages.append({
                    "role": "user",
                    "content": (
                        f"🚨 NEGOTIATION STALLED — {chatter_streak} consecutive iterations "
                        "with only send_message/ping_peer. This conversation is going in circles.\n\n"
                        "You MUST break the loop RIGHT NOW. Exactly two choices:\n\n"
                        "CHOICE A — Commit to the patch:\n"
                        "  Call grep_file to get the exact verbatim current line from the file,\n"
                        "  then in the SAME response call send_message with the message:\n"
                        "    'PATCH PROPOSAL: old_text=<exact text from file> "
                        "new_text=<exact replacement>'\n"
                        "  Use the verbatim text — no paraphrasing, no pseudocode.\n\n"
                        "CHOICE B — Abandon and pick a different change:\n"
                        "  Call grep_file with a NEW pattern to read a DIFFERENT constant,\n"
                        "  then send_message proposing that different change instead.\n\n"
                        "Do NOT send another 'Agreed, let's discuss' message. "
                        "Do NOT call ping_peer. That is not a choice."
                    ),
                })
            elif chatter_streak >= 3:
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠ CHATTER LOOP DETECTED — {chatter_streak} iterations of only "
                        "send_message with no file reads or patches.\n\n"
                        "You and your peer are agreeing without acting. This must stop.\n\n"
                        "Your next response MUST include grep_file to read the exact "
                        "current line you want to change. Without seeing the verbatim text "
                        "from the file, you cannot produce a valid patch_own_file call.\n\n"
                        "Call grep_file NOW with a specific pattern that finds the exact "
                        "line. Then send_message stating the exact old_text= and new_text= "
                        "you propose — not a description, the LITERAL strings."
                    ),
                })

        else:
            # No tool called
            no_action_streak += 1
            peer_alive = _silent_ping()
            peer_status = (
                f"Your peer (Instance {3 - INSTANCE_NUM}) is currently "
                + ("ONLINE." if peer_alive else
                   "NOT responding (may still be loading — model takes ~30s).")
            )
            if no_action_streak >= 3:
                nudge = (
                    f"{peer_status}\n"
                    "You haven't taken an action in several iterations. "
                    "If waiting for a reply: use ping_peer to confirm they're alive, "
                    "or send another message. If you have no pending work, describe "
                    "what you're waiting for."
                )
            else:
                nudge = (
                    f"{peer_status}\n"
                    "No tool was called. What is your next action?"
                )
            messages.append({"role": "user", "content": nudge})

        # Keep conversation history manageable
        if len(messages) > 32:
            messages = [messages[0]] + messages[-30:]

        time.sleep(1)

    logging.info("Maximum iterations reached. Instance stopping.")

# ---------------------------------------------------------------------------
# Initial Setup — create directories and default tools
# ---------------------------------------------------------------------------

def setup_environment():
    """Create the working directories and default tool files."""
    STATE_DIR.mkdir(exist_ok=True)
    TOOLS_DIR.mkdir(exist_ok=True)
    (STATE_DIR / "inbox").mkdir(exist_ok=True)

    # If we're starting up after a restart_self, clear the flag NOW — before
    # the HTTP listener starts — so our peer's next ping sees us as fully up.
    restart_flag = STATE_DIR / "restarting.flag"
    if restart_flag.exists():
        restart_flag.unlink()
        logging.info("Cleared restarting.flag — restart complete.")

    auth_flag = STATE_DIR / "patch_authorized.flag"
    if auth_flag.exists():
        auth_flag.unlink(missing_ok=True)
        logging.info("Cleared stale patch_authorized.flag at startup.")

    default_discover = TOOLS_DIR / "discover_hosts.sh"
    if not default_discover.exists():
        default_discover.write_text("""#!/bin/bash
# Discover hosts on the local network
echo "=== Network Interfaces ==="
ifconfig | grep "inet " | grep -v 127.0.0.1
echo ""
echo "=== ARP Table (known hosts) ==="
arp -a
echo ""
echo "=== Checking common SSH ports ==="
for ip in $(arp -a | grep -oE '\\b[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\b'); do
    nc -z -w 1 $ip 22 2>/dev/null && echo "SSH open: $ip"
done
""")
        os.chmod(default_discover, 0o755)

    default_replicate = TOOLS_DIR / "replicate.sh"
    if not default_replicate.exists():
        default_replicate.write_text("""#!/bin/bash
# Replication helper
REMOTE_HOST=$1
REMOTE_USER=${2:-$(whoami)}
REMOTE_DIR="~/emergence"
if [ -z "$REMOTE_HOST" ]; then
    echo "Usage: $0 <remote_host> [remote_user]"
    exit 1
fi
echo "=== Replicating to $REMOTE_USER@$REMOTE_HOST ==="
ssh -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH OK'" || exit 1
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR/tools $REMOTE_DIR/state"
scp emergence.py "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
scp -r tools/ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && pip install mlx-lm && nohup python3 emergence.py > state/startup.log 2>&1 &"
echo "=== Done ==="
""")
        os.chmod(default_replicate, 0o755)

    meta = {
        "instance_id": INSTANCE_ID,
        "instance_num": INSTANCE_NUM,
        "host": socket.gethostname(),
        "port": LISTEN_PORT,
        "peer_port": PEER_PORT,
        "working_dir": str(WORKING_DIR),
        "born": datetime.now(timezone.utc).isoformat(),
        "model": _MODEL_NAME,
        "pid": os.getpid(),
    }
    (STATE_DIR / "identity.json").write_text(json.dumps(meta, indent=2))
    logging.info(f"Environment ready. Tools: {list(TOOLS_DIR.glob('*'))}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    global LISTEN_PORT, INSTANCE_NUM, PEER_PORT, PEER_DIR, WORKING_DIR
    global STATE_DIR, TOOLS_DIR, LOG_FILE, _MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Emergence — Self-replicating LLM orchestrator demo"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"MLX model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Maximum loop iterations (default: {MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--port", type=int, default=LISTEN_PORT,
        help=f"My communication port (default: {LISTEN_PORT})"
    )
    parser.add_argument(
        "--peer-port", type=int, default=7701,
        help="Peer instance port (default: 7701)"
    )
    parser.add_argument(
        "--instance-num", type=int, default=1, choices=[1, 2],
        help="Instance number: 1=parent (default), 2=child"
    )
    parser.add_argument(
        "--working-dir", type=str, default=".",
        help="Working directory for this instance (default: current dir)"
    )
    parser.add_argument(
        "--peer-dir", type=str, default="",
        help="Peer instance's working directory (enables cross-instance writes)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print system prompt and exit (no model loading)"
    )
    args = parser.parse_args()

    # Set all globals before anything else
    LISTEN_PORT  = args.port
    PEER_PORT    = args.peer_port
    INSTANCE_NUM = args.instance_num
    WORKING_DIR  = Path(args.working_dir).resolve()
    _MODEL_NAME  = args.model

    if args.peer_dir:
        PEER_DIR = Path(args.peer_dir).resolve()

    STATE_DIR = WORKING_DIR / "state"
    TOOLS_DIR = WORKING_DIR / "tools"
    LOG_FILE  = STATE_DIR / "emergence.log"

    # ── Instance 1 self-relocation ─────────────────────────────────────────────
    # On a fresh start, I1 copies itself to /tmp so every run begins with a
    # clean state/ directory — no stale restart_goal.json, inbox, or memory
    # carried over from a previous session.
    # Detection: if __file__ is already under /tmp, this is either a restart
    # (tool_restart_self re-execs from /tmp) or a relocation already happened —
    # skip.  Dry-run also skips so the user sees the prompt from the source dir.
    # On macOS /tmp is a symlink → /private/tmp, so resolve both before comparing.
    _tmp_real = str(Path('/tmp').resolve())  # /private/tmp on macOS, /tmp elsewhere
    if INSTANCE_NUM == 1 and not str(Path(__file__).resolve()).startswith(_tmp_real) and not args.dry_run:
        i1_short = generate_instance_id()[:8]
        tmp_dir = Path(f"/tmp/emergence_1_{i1_short}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        src_script = Path(__file__).resolve()
        shutil.copy2(src_script, tmp_dir / "emergence.py")
        src_tools = src_script.parent / "tools"
        if src_tools.exists():
            dst_tools = tmp_dir / "tools"
            if dst_tools.exists():
                shutil.rmtree(dst_tools)
            shutil.copytree(str(src_tools), str(dst_tools))
        new_argv = [
            sys.executable, str(tmp_dir / "emergence.py"),
            "--working-dir", str(tmp_dir),
            "--port", str(LISTEN_PORT),
            "--peer-port", str(PEER_PORT),
            "--instance-num", "1",
            "--model", _MODEL_NAME,
            "--max-iterations", str(args.max_iterations),
        ]
        if args.peer_dir:
            new_argv += ["--peer-dir", args.peer_dir]
        print(f"[I1] Relocating to {tmp_dir} for clean startup ...", flush=True)
        os.execv(sys.executable, new_argv)
        # os.execv replaces this process — code below never runs

    setup_logging()
    setup_environment()

    if args.dry_run:
        print("=" * 60)
        print("SYSTEM PROMPT (what the LLM sees):")
        print("=" * 60)
        print(get_system_prompt())
        print("=" * 60)
        print(f"\nTools available: {list(_build_tools().keys())}")
        print(f"Instance ID    : {INSTANCE_ID}")
        print(f"Working dir    : {WORKING_DIR}")
        return

    server = start_listener()
    if server is None:
        logging.error(
            f"Could not bind port {LISTEN_PORT} after retries — exiting. "
            "Another instance is likely already running on this port."
        )
        sys.exit(1)
    _start_heartbeat()

    llm = LocalLLM(model_name=args.model)
    llm.load()

    try:
        run_agentic_loop(llm, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        logging.info("Stopped by user (Ctrl+C). Instance terminating.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
