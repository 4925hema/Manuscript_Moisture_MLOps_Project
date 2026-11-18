"""One-command reproducible pipeline: create data -> feature store -> train/register."""
from .generate_data import main as generate
from .feature_pipeline import main as materialize
from .train import main as train


def main() -> None:
    print("[1/3] Generating Version 1 and Version 2 data")
    generate()
    print("[2/3] Building and materializing engineered features")
    materialize()
    print("[3/3] Training, comparing and registering the best model")
    train()


if __name__ == "__main__":
    main()
