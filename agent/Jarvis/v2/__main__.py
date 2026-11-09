"""Run Jarvis v2 as a package or directly from an IDE."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent.Jarvis.v2.code import main
else:
    from .code import main


if __name__ == "__main__":
    main()
