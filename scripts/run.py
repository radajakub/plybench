# kept so `uv run python scripts/run.py ...` still works in this checkout; the tool is plybench.cli.run, installed as plybench-run
from plybench.cli.run import main

if __name__ == "__main__":
    main()
