# kept so `uv run python scripts/analyze.py ...` still works in this checkout; the tool is plybench.cli.analyze, installed as plybench-analyze
from plybench.cli.analyze import main

if __name__ == "__main__":
    main()
