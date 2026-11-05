# kept so `uv run python scripts/play.py ...` still works in this checkout; the tool is plybench.cli.play, installed as plybench-play
from plybench.cli.play import main

if __name__ == "__main__":
    main()
