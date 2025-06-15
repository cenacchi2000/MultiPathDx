import argparse
import os
from multipathdx.dataset import EDAICDataset
from multipathdx.orchestrator import MultiPathDx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MultiPathDx on E-DAIC")
    parser.add_argument("data_root", help="Path to the E-DAIC dataset root")
    parser.add_argument("split_file", help="CSV file with training split and labels")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    dataset = EDAICDataset(args.data_root, args.split_file)
    system = MultiPathDx(dataset, lr=args.lr, epochs=args.epochs)
    system.train()
    acc = system.evaluate()
    print(f"Training accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
