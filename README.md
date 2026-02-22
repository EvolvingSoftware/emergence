# Emergence

**Two local LLM instances that collaborate to modify their own source code, restart with the changes, and repeat, indefinitely.**

This is a working demonstration of the ideas in [Evolving Software, The Emergence Architecture](https://evolvingsoftware.com): software that exhibits evolutionary behaviour through constraint, replication, variation, feedback, and communication, with no external orchestration and no cloud.

## What It Does

`emergence.py` runs two instances of itself on your Mac. Each instance is powered by a local 7B LLM. The two instances communicate over localhost, reason about the codebase, and take turns applying improvements to their own source code.

**One cycle looks like this:**

1. **Both instances read** `emergence.py` independently and identify candidate improvements across three categories: Configuration, Reliability, or Capability.
2. **They discuss** via HTTP messaging, converging on one proposal per cycle.
3. **Instance 2 patches first** (canary) and restarts with the new code.
4. **Instance 1 verifies** the patch is live in Instance 2's running file.
5. **Instance 1 patches** its own copy and restarts.
6. Both instances come back online with the improved code and begin the next cycle.

The LLM is not following a patch script. It reads the real file, reasons about what to change and why, applies the change as a text patch, verifies it, and restarts itself. That's the point.

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.10+**
- **~6 GB free RAM** (for the Qwen 2.5 7B 4-bit model)

No second machine. No cloud account. No API key.

## Quick Start

```bash
pip install mlx-lm
python3 emergence.py
```

The first run downloads the model (~4.5 GB from HuggingFace) and caches it locally. After that it runs fully offline.

Instance 1 starts, reads its own source, proposes an improvement, then spawns Instance 2 in a new terminal window. Both instances negotiate, patch, and restart automatically.

## Watching It Run

```bash
# All inter-instance messages in real time
tail -f /tmp/emergence_comm.log

# Full activity log for Instance 1
tail -f state/emergence.log
```

To see the system prompt and tool list without loading the model:

```bash
python3 emergence.py --dry-run
```

## Architecture

```
emergence.py                         ← Orchestrator + agentic loop. Self-modifiable.
tools/
  discover_hosts.sh                  ← Network discovery (cross-machine replication)
  replicate.sh                       ← SSH-based replication helper
state/
  identity.json                      ← This instance's metadata
  emergence.log                      ← Full activity log
  inbox/                             ← Incoming messages from the peer instance
  restart_goal.json                  ← Patch context, persisted across restarts
/tmp/
  emergence_comm.log                 ← All inter-instance messages (both directions)
  emergence_instance_N_<id>.log      ← Per-instance detailed log
```

### Self-Modification

Each instance can:

- **`patch_own_file`** : apply a targeted text patch to its own `emergence.py` (auto-backs up to `.bak` first)
- **`restore_own_file`** : roll back to the backup if a patch breaks something
- **`restart_self`** : relaunch with the updated code; the patch goal is persisted so the instance picks up where it left off
- **`write_file`** : create new tool scripts in `tools/` without touching `emergence.py`

Turn-taking is enforced in Python: Instance 2 always patches first (canary); Instance 1 patches second after verification. Never both at once.

### Three Improvement Categories

Each cycle the instances evaluate three candidates, one from each category where possible:

| Category | Targets |
|----------|---------|
| **A — Configuration** | Constants and numeric thresholds (timeouts, caps, buffer sizes) |
| **B — Reliability** | Error handling, retry logic, protocol robustness, bad-patch detection |
| **C — Capability** | New or improved tools, smarter scheduling, new scripts in `tools/` |

Category A is limited to one proposal per cycle. Categories B and C are preferred for subsequent cycles.

### Communication

Instances communicate via HTTP on `localhost:7700` (Instance 1) and `localhost:7701` (Instance 2). Messages are written to `state/inbox/` so they survive restart gaps. The comm log at `/tmp/emergence_comm.log` shows the full conversation between instances.

### The Living-Off-the-Land Philosophy

The orchestrator is minimal. It gives the LLM the ability to read and patch its own code, run shell commands, and communicate with its peer. Everything else — what to improve, why, how to verify it landed; the LLM reasons through itself.

The `tools/` directory contains starter scripts, but instances can rewrite or extend them. A tool that doesn't work well is like a limb that doesn't flex properly — the organism adapts.

## Model Choice

Default: `mlx-community/Qwen2.5-7B-Instruct-4bit`

- ~4.5 GB on disk and in RAM
- Strong instruction-following and code reasoning
- ~28–35 tokens/sec on M-series Macs with 16 GB RAM
- Leaves ~10 GB free for macOS and other processes

For Macs with 8 GB RAM:

```bash
python3 emergence.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

## Cross-Machine Replication

The original capability is still present. If Instance 1 detects another Mac on the network with SSH enabled, it can replicate: scan, SSH in, install dependencies, copy the file, and start a new instance. The child runs independently and can join the improvement cycle.

To enable this:

1. On the target Mac: System Settings → General → Sharing → Remote Login → On
2. Ensure Python 3.10+ is installed on the target

The LLM discovers the host via `scan_network`, negotiates SSH access, and handles the transfer. You don't script it, it figures it out.

## Safety

- **Iteration cap** : the loop stops after 200 cycles by default.
- **Kill switch** : Ctrl+C stops any instance immediately.
- **Canary deployment** : Instance 2 always patches first. If it breaks, Instance 1 can roll back via `restore_peer_file` before patching itself.
- **Auto-backup** : every patch saves `.bak` automatically; `restore_own_file` rolls back instantly.
- **Write restrictions** : `write_file` is restricted to `tools/` and `state/`; changes to `emergence.py` must go through `patch_own_file`.
- **Full logging** : every tool call, shell command, and LLM decision is logged.
- **Chatter detection** : the loop breaks idle back-and-forth between instances that produces no real action.
- **Hallucination detection** : responses that begin with peer-message markers are discarded and the instance is corrected.
- **No open weights online** : the model runs locally; there is no external call to make or intercept.
- **Manual SSH required for cross-machine spread** : replication only works if you've explicitly enabled Remote Login on the target.

## Mapping to the Seven Layers

| Layer | How It Appears in the Demo |
|-------|---------------------------|
| I. Constraint | RAM limits, iteration caps, port boundaries, turn-taking rules |
| II. Self-Replication | Instances restart with updated code; cross-machine replication copies to new hosts |
| III. Variation | Each patch cycle introduces a distinct change; successive cycles can diverge |
| IV. Feedback | Each instance observes the result of its patches and adjusts in the next cycle |
| V. Influence Without Deletion | Improvements discovered by one instance propagate to the other through the shared patch protocol |
| VI. Temporal Compression | The LLM reasons and acts in seconds, not months |
| VII. Cascading Interdependence | Instances depend on each other's verification before applying changes |

## The Paper

This demo accompanies *Evolving Software, The Emergence Architecture: Why Software Does Not Need Sentience to Surpass Us*, available at [EvolvingSoftware.com](https://evolvingsoftware.com). A PDF copy of the paper is available here: [EvolvingSoftwareTheEmergenceArchitecture.pdf](EvolvingSoftwareTheEmergenceArchitecture.pdf)

The paper presents a seven-layer framework for understanding how software can evolve without consciousness and argues that the structural conditions for this are already assembling.

## License

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
