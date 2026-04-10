"""
visualize.py — CLI entry point for the Planck log visualizer.

Usage:
    python visualizer/visualize.py run.log
    python visualizer/visualize.py run.log --port 8051
    python -m visualizer run.log
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="visualize",
        description=(
            "Launch the Planck interactive visualizer for a calculation log."
        ),
    )
    parser.add_argument(
        "log_file",
        help="Path to the Planck stdout log file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        metavar="INT",
        help="Port to serve on (default: 5050).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="STR",
        help="Host address to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode.",
    )
    args = parser.parse_args()

    # Parse log
    from visualizer.parser import parse_log
    try:
        run = parse_log(args.log_file)
    except Exception as exc:
        print(
            f"error: could not parse log file '{args.log_file}': {exc}",
            file=sys.stderr,
        )
        return 1

    # Build Flask app
    from visualizer.app import create_app
    flask_app = create_app(run, log_path=args.log_file)

    # Startup banner
    url = f"http://{args.host}:{args.port}"
    n_opt  = len(run.opt_steps)
    n_freq = len(run.freq_modes)
    n_traj = len(run.opt_step_geometries)
    n_cas  = len(run.casscf_iters)
    sep = "  ─────────────────────────────────────────"
    traj_note = f"  ({n_traj} with geometry)" if n_traj else ""
    cas_note = (
        f"  ({run.casscf_active_space}"
        f"{', SA(' + str(run.casscf_n_roots) + ')' if run.casscf_n_roots else ''})"
        if n_cas else ""
    )
    lines = [
        "",
        "  Planck Log Viewer",
        sep,
        f"  Log   : {Path(args.log_file).resolve()}",
        f"  Calc  : {run.calculation_type or '—'}",
        f"  Level : {run.scf_type or '—'}  /  {run.basis or '—'}",
        f"  Group : {run.point_group or '—'}",
        f"  SCF   : {len(run.scf_iters)} iterations",
        (f"  CAS   : {n_cas} macro iterations{cas_note}" if n_cas
         else "  CAS   : —"),
        f"  Opt   : {n_opt} steps{traj_note}",
        f"  Freq  : {n_freq} modes",
        sep,
        f"  URL   : {url}",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)

    if not args.no_browser:
        threading.Timer(1.0, _open_browser, args=(url,)).start()

    flask_app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
