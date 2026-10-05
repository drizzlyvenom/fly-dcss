"""Run a bounded local DCSS 0.17.1 episode (Linux/macOS Unix sockets + PTY)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pty
import resource
import sys
import signal
import struct
import subprocess
import tempfile
import threading
import time
import fcntl
import termios

import numpy as np
from .agent import LoopAgent
from .demo import synthetic_graph
from .webtiles import LocalConnection, PipeConnection, UnsafeInput


def write_record(stream, kind, **fields):
    stream.write(json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n")
    stream.flush()


def finite_chunks(array):
    return all(np.isfinite(array[start:start + 1_000_000]).all()
               for start in range(0, len(array), 1_000_000))


def run(args):
    if not 1 <= args.steps <= 1000 or not 1 <= args.seconds <= 600:
        raise ValueError("steps must be 1..1000 and seconds 1..600")
    binary = Path(args.crawl).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation_started = time.monotonic()
    graph = synthetic_graph()
    provenance = {"kind": "synthetic", "biological": False, "description": "invented three-neuron fixture"}
    if getattr(args, "malecns", None):
        from .malecns import load_malecns
        graph, provenance = load_malecns(args.malecns)
    elif args.connectome:
        from .connectome import load_flyhero
        graph, provenance = load_flyhero(args.connectome, n=args.neurons)
    agent = LoopAgent(graph, seed=args.seed, compact=bool(getattr(args, "malecns", None)))
    version = subprocess.run([str(binary), "-version"], capture_output=True, text=True, timeout=10)
    version_text = version.stdout + version.stderr
    if version.returncode or "Crawl version 0.17.1" not in version_text.splitlines():
        raise ValueError(f"requires verified DCSS 0.17.1 executable: {version_text[:300]}")
    started = time.monotonic()
    deadline = started + args.seconds
    cumulative_update_l1 = 0.0
    process = connection = None
    master = slave = None
    pipe_fds = []
    completed = 0
    stop = "error"
    with (output / "episode.jsonl").open("w") as log, tempfile.TemporaryDirectory(prefix="fdcss-") as temporary:
        temp = Path(temporary)
        rc = temp / "init.txt"
        rc.write_text("name = FlyLoop\nspecies = Human\nbackground = Fighter\nweapon = long sword\nshow_more = false\ntile_skip_title = true\nrestart_after_game = false\nautopickup =\n")
        command = [str(binary), "-rc", str(rc), "-dir", str(output), "-name", "FlyLoop", "-species", "Human", "-background", "Fighter", "-seed", format(args.seed, "x"), "-no-save", "-webtiles-socket", str(temp / "game.sock"), "-await-connection"]
        write_record(log, "metadata", game="DCSS 0.17.1", version=version_text.strip(),
                     binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                     command=command, transport=args.transport, seed=args.seed, max_actions=args.steps, max_seconds=args.seconds,
                     graph=provenance, nodes=graph.n, edges=len(graph.weight),
                     parameters=vars(agent.circuit.parameters),
                     dtype=str(agent.circuit.weights.dtype),
                     initial_weights_min=float(agent.circuit.weights.min()),
                     initial_weights_max=float(agent.circuit.weights.max()))
        try:
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
            env = {**os.environ, "TERM": "xterm"}
            pass_fds = ()
            if args.transport == "pipe":
                child_read, parent_write = os.pipe()
                parent_read, child_write = os.pipe()
                pipe_fds = [child_read, parent_write, parent_read, child_write]
                env.update(FLY_DCSS_READ_FD=str(child_read), FLY_DCSS_WRITE_FD=str(child_write))
                pass_fds = (child_read, child_write)
            process = subprocess.Popen(command, cwd=binary.parent, stdin=slave, stdout=slave, stderr=slave,
                                       env=env, pass_fds=pass_fds, start_new_session=True)
            if args.transport == "pipe":
                os.close(child_read)
                os.close(child_write)
                pipe_fds = [parent_read, parent_write]
            os.close(slave)
            slave = None
            def drain():
                total = 0
                with (output / "console.log").open("wb") as stream:
                    while True:
                        try:
                            data = os.read(master, 8192)
                        except OSError:
                            break
                        if not data:
                            break
                        if total < 2 * 1024 * 1024:
                            stream.write(data)
                        total += len(data)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            if args.transport == "pipe":
                connection = PipeConnection(parent_read, parent_write, process)
                pipe_fds = []  # connection now owns these descriptors
            else:
                while not (temp / "game.sock").exists():
                    if process.poll() is not None:
                        raise RuntimeError(f"DCSS startup failed ({process.returncode}); see console.log")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("DCSS startup deadline exceeded")
                    time.sleep(0.02)
                connection = LocalConnection(temp / "game.sock", temp / "client.sock", process)
            connection.attach()
            connection.receive_frame(deadline)
            if connection.state.version != "0.17.1":
                raise ValueError(f"unexpected protocol version {connection.state.version!r}")
            # Initial deltas omit zero-valued HUD fields; request the native full snapshot.
            connection.send({"msg": "spectator_joined"})
            connection.receive_frame(deadline)
            observation = connection.state.snapshot()
            agent.external(observation)
            write_record(log, "mapping", **agent.mapping_summary())
            write_record(log, "initial_observation", observation=observation)
            for index in range(args.steps):
                if time.monotonic() >= deadline:
                    raise TimeoutError("episode deadline exceeded")
                action, decision = agent.choose(observation)
                if time.monotonic() >= deadline:
                    raise TimeoutError("episode deadline exceeded during circuit choice")
                write_record(log, "action", index=index, turn=observation["player"]["turn"], action=action, decision=decision)
                connection.act(action)
                connection.receive_frame(deadline)
                after = connection.state.snapshot()
                result = agent.feedback(observation, after)
                cumulative_update_l1 += result["weights_delta_l1"]
                write_record(log, "transition", index=index, action=action, observation=after, **result)
                completed += 1
                observation = after
            stop = "action_limit"
        except (UnsafeInput, TimeoutError, RuntimeError, ValueError, OSError) as error:
            stop = type(error).__name__
            write_record(log, "error", error=str(error), mode=connection.state.mode if connection else None,
                         ui=connection.state.ui if connection else None)
        finally:
            # Terminate this isolated no-save child, never send quit keys to an unknown prompt.
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGHUP)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            if connection:
                connection.close()
            for descriptor in pipe_fds:
                os.close(descriptor)
            if slave is not None:
                os.close(slave)
            if master is not None:
                os.close(master)
            net_delta_l1 = 0.0
            for start in range(0, len(graph.weight), 1_000_000):
                sl = slice(start, start + 1_000_000)
                baseline = np.clip(np.multiply(graph.weight[sl], agent.circuit.parameters.weight_scale, dtype=np.float64),
                                   0, agent.circuit.parameters.weight_max).astype(agent.circuit.dtype)
                net_delta_l1 += float(np.abs(np.subtract(agent.circuit.weights[sl], baseline, dtype=np.float64)).sum(dtype=np.float64))
            checkpoint = output / "circuit-final.npz"
            np.savez(checkpoint, state=agent.circuit.state, weights=agent.circuit.weights,
                     eligibility=agent.circuit.eligibility, time=agent.circuit.time)
            summary = {"nodes": graph.n, "edges": len(graph.indices),
                       "mapping": agent.mapping_summary(),
                       "preparation_and_episode_seconds": time.monotonic() - preparation_started,
                       "checkpoint": checkpoint.name, "checkpoint_resumable_game": False,
                       "state_finite": finite_chunks(agent.circuit.state),
                       "weights_finite": finite_chunks(agent.circuit.weights),
                       "eligibility_finite": finite_chunks(agent.circuit.eligibility),
                       "weights_cumulative_delta_l1": cumulative_update_l1,
                       "completed_actions": completed, "stop_reason": stop, "output": str(output),
                       "data_kind": provenance.get("kind", "connectome-derived"), "learning_efficacy_tested": False,
                       "elapsed_seconds": time.monotonic() - started,
                       "final_turn": connection.state.player.get("turn") if connection else None,
                       "weights_net_delta_l1": net_delta_l1,
                       "weights_min": float(agent.circuit.weights.min()),
                       "weights_max": float(agent.circuit.weights.max())}
            summary["python_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
            write_record(log, "summary", **summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl", required=True, help="locally built exact 0.17.1 WebTiles executable")
    parser.add_argument("--transport", choices=("pipe", "unix"), default="pipe")
    parser.add_argument("--output", required=True, help="new directory for logs")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=7)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--malecns", help="full prepared MaleCNS array directory")
    source.add_argument("--connectome", help="pinned fly-hero subgraph JSON; omitted means synthetic")
    parser.add_argument("--neurons", type=int, default=64)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["stop_reason"] == "action_limit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
