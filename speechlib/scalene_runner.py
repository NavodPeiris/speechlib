"""
Scalene runner — line-level CPU / GPU / memory profiler.

Launches any script under scalene, which profiles line-by-line without
requiring code changes. Separates CPU compute, GPU time, and memory
allocation per source line.

Usage:
    python -m speechlib.scalene_runner process_audio.py [script args...]
    python -m speechlib.scalene_runner --html process_audio.py [script args...]

Flags:
    --html   Write HTML report to scalene_report.html (default: terminal output)
    --help   Show this message

What scalene shows:
    - % CPU time per line (Python vs native/C)
    - % GPU time per line
    - Memory allocation per line (CPU heap + GPU)
    - I/O wait is separated automatically from CPU compute
    - ~15% overhead (use for debugging, not production)

Install scalene:
    pip install scalene
"""
import subprocess
import sys


def run(script: str, *script_args: str, html_output: str | None = None) -> int:
    """
    Run `script` under scalene and return the exit code.

    Args:
        script:      Path to the Python script to profile.
        *script_args: Arguments forwarded to the script.
        html_output: If set, write HTML report to this path instead of terminal.
    """
    cmd = [sys.executable, "-m", "scalene", "--gpu"]
    if html_output:
        cmd += ["--html", "--outfile", html_output]
    cmd += ["--", script, *script_args]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def _check_scalene() -> bool:
    try:
        import scalene  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    args = sys.argv[1:]

    if not args or "--help" in args:
        print(__doc__)
        sys.exit(0)

    if not _check_scalene():
        print("scalene is not installed. Run:  pip install scalene")
        sys.exit(1)

    html_output = None
    if "--html" in args:
        args = [a for a in args if a != "--html"]
        html_output = "scalene_report.html"

    if not args:
        print("Error: no script specified.")
        sys.exit(1)

    script, *script_args = args
    sys.exit(run(script, *script_args, html_output=html_output))


if __name__ == "__main__":
    main()
