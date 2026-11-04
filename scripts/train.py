# kept so `uv run python scripts/train.py ...` still works in this checkout; the tool is plybench.cli.train, installed as plybench-train
from plybench.cli.train import main

if __name__ == "__main__":
    main()
