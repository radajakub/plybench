# kept so `uv run python scripts/plot.py ...` still works in this checkout; the tool is plybench.cli.plot, installed as plybench-plot
from plybench.cli.plot import main

if __name__ == "__main__":
    main()
