"""Unified Antigravity Agent Master CLI.
Supports:
  agent run "<task>"
  agent run "<task>" --auto
  agent run "<task>" --dry-run
  agent resume <task_id>
  agent stop
  agent purge
  agent call
  agent terminal "<command>"
  agent trusted-session start|end
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy.gateway import PolicyGateway, RiskTier, InputProvenance
from agent.core.wal import WALManager
from agent.core.execution_engine import ExecutionEngine
from agent.core.auto_mode import AutoController, AutoLimits
from agent.tools.terminal import TerminalSubsystem
from agent.voice.pipeline import VoicePipeline
from agent.supervisor.watchdog import ProcessSupervisor


def get_default_engine():
    fixture_dir = str(PROJECT_ROOT / "eval" / "fixtures")
    wal_db = str(Path(fixture_dir) / "cli_wal.db")
    gateway = PolicyGateway(granted_roots=[fixture_dir, str(PROJECT_ROOT)])
    wal = WALManager(db_path=wal_db)
    return ExecutionEngine(gateway=gateway, wal=wal, fixture_dir=fixture_dir), gateway, wal


def cmd_run(args):
    print(f"[*] Task Received: '{args.task}'")
    engine, gateway, wal = get_default_engine()

    if args.dry_run:
        print("[DRY-RUN MODE] Analyzing task without executing OS actions...")
        charter = engine.understand("task_dry_run", args.task)
        plan = engine.create_plan_for_workflow("task_dry_run", "preview_report.txt", "Dry-run preview")
        print(f"[DRY-RUN] Planned {len(plan)} steps:")
        for s in plan:
            print(f"  - Step {s.step_index} [{s.risk_tier.value}]: {s.intent} (Tool: {s.tool})")
        print("[DRY-RUN] Complete. Zero mutations executed.")
        return

    if args.auto:
        print("[AUTO MODE] Launching bounded autonomous controller...")
        controller = AutoController(engine, limits=AutoLimits(max_steps=10, max_wall_clock_seconds=300.0))
        t_id = f"task_{int(time.time())}"
        res = controller.run_bounded_task(
            task_id=t_id,
            prompt=args.task,
            target_filename=f"report_{t_id}.txt",
            report_content=f"Report for task: {args.task}\nCompleted autonomously.\n",
        )
        print(f"[AUTO MODE] Finished. Result: {res}")
        return

    # Standard run
    t_id = f"task_{int(time.time())}"
    res = engine.run_workflow(
        task_id=t_id,
        prompt=args.task,
        target_filename=f"report_{t_id}.txt",
        report_text=f"Task findings for: {args.task}\nVerified execution report.\n",
    )
    print(f"[*] Task Completed: {res}")


def cmd_resume(args):
    print(f"[*] Reconciling WAL after crash for task: '{args.task_id}'...")
    engine, gateway, wal = get_default_engine()
    res = engine.reconcile_on_resume(args.task_id)
    print(f"[*] Reconciliation Result: {res}")


def cmd_stop(args):
    print("[*] Triggering out-of-process Emergency Stop...")
    sup = ProcessSupervisor()
    latency = sup.emergency_stop()
    print(f"[*] Emergency stop signal executed in {latency*1000:.2f} ms.")


def cmd_purge(args):
    print("[*] Purging temporary flight logs and cached artifacts...")
    fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
    purged_count = 0
    if fixture_dir.exists():
        for p in fixture_dir.glob("*.tmp*"):
            try:
                p.unlink()
                purged_count += 1
            except Exception:
                pass
    print(f"[*] Purged {purged_count} temporary files.")


def cmd_terminal(args):
    engine, gateway, wal = get_default_engine()
    term = TerminalSubsystem(gateway=gateway)
    print(f"[*] Running terminal command: '{args.command}'")
    res = term.execute_command(args.command)
    print(f"[Exit: {res.exit_code}] (Duration: {res.duration_seconds}s)")
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(f"[STDERR]\n{res.stderr}")


def cmd_call(args):
    from llm_engine import get_llm_engine

    llm_engine = get_llm_engine()

    # Load model synchronously before starting the call so every turn gets a real reply.
    print("[*] Loading language model... (this may take a moment on first run)")
    try:
        llm_engine.load_model()
        print("[*] Model loaded. Starting voice call.")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("[*] Proceeding in degraded mode – replies will be limited.")

    def _llm_gen(user_query: str) -> str:
        """Bridge from VoicePipeline to the loaded LLM engine."""
        if not llm_engine.is_loaded:
            return "The language model is still loading. Please wait a moment and try again."
        try:
            res = llm_engine.generate_answer(
                question=user_query,
                max_tokens=256,
                temperature=0.7,
            )
            return res.get("reply") or "I heard you but could not generate a reply."
        except Exception as exc:
            return f"I encountered an error generating a response: {exc}"

    pipeline = VoicePipeline()
    pipeline.run_live_call_session(llm_generate_fn=_llm_gen)


def cmd_voice_doctor(args):
    from agent.voice.doctor import VoiceDoctor
    doc = VoiceDoctor(record_seconds=2.0, enable_playback=True)
    success = doc.run_all()
    sys.exit(0 if success else 1)


def cmd_trusted_session(args):
    engine, gateway, wal = get_default_engine()
    term = TerminalSubsystem(gateway=gateway)
    if args.action == "start":
        ts = term.start_trusted_session("cli_session", str(PROJECT_ROOT), duration_seconds=1800.0)
        print(f"[*] Trusted Terminal Session started: '{ts.session_id}' in '{ts.allowed_root_directory}' (Expires in 30m).")
    else:
        term.end_trusted_session()
        print("[*] Trusted Terminal Session terminated.")


def main():
    parser = argparse.ArgumentParser(description="Antigravity Master Agent CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # run
    p_run = subparsers.add_parser("run", help="Run a task")
    p_run.add_argument("task", type=str, help="Goal or instruction prompt")
    p_run.add_argument("--auto", action="store_true", help="Enable bounded Auto Mode")
    p_run.add_argument("--dry-run", action="store_true", help="Preview plan without executing OS actions")
    p_run.set_defaults(func=cmd_run)

    # resume
    p_res = subparsers.add_parser("resume", help="Resume task after crash")
    p_res.add_argument("task_id", type=str, help="ID of task to resume")
    p_res.set_defaults(func=cmd_resume)

    # stop
    p_stop = subparsers.add_parser("stop", help="Emergency stop")
    p_stop.set_defaults(func=cmd_stop)

    # purge
    p_purge = subparsers.add_parser("purge", help="Purge temporary artifacts")
    p_purge.set_defaults(func=cmd_purge)

    # call
    p_call = subparsers.add_parser("call", help="Launch real-time voice call mode")
    p_call.set_defaults(func=cmd_call)

    # voice-doctor
    p_doc = subparsers.add_parser("voice-doctor", help="Run stage-by-stage voice pipeline health probe")
    p_doc.set_defaults(func=cmd_voice_doctor)

    # terminal
    p_term = subparsers.add_parser("terminal", help="Execute policy-governed terminal command")
    p_term.add_argument("command", type=str, help="Command string")
    p_term.set_defaults(func=cmd_terminal)

    # trusted-session
    p_ts = subparsers.add_parser("trusted-session", help="Manage trusted terminal sessions")
    p_ts.add_argument("action", choices=["start", "end"], help="Start or end session")
    p_ts.set_defaults(func=cmd_trusted_session)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
