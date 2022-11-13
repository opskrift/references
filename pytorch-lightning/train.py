import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from model import MNISTModel

# add top level src/ dir to path
# TODO: maybe extract useful code into a package?
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_utils import args_path_ensure_exists


def main():
    if cli_args.seed is not None:
        pl.utilities.seed.seed_everything(cli_args.seed, workers=True)

    model = MNISTModel(
        data_dir=cli_args.dataset_path,
        hidden_size=cli_args.hidden_size,
        learning_rate=cli_args.learning_rate,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )
    print(model)

    tb_logger = pl.loggers.TensorBoardLogger(
        "tb_logs", name="mnist", default_hp_metric=False
    )

    trainer = pl.Trainer(
        gpus=(1 if cli_args.cuda else 0),
        max_epochs=cli_args.max_epochs,
        progress_bar_refresh_rate=20,
        logger=tb_logger,
    )
    print(trainer)

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument("--hidden_size", type=int, required=False, default=64)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--dataset_path", type=args_path_ensure_exists, required=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Whether to use CUDA for training."
    )
    cli_args = parser.parse_args()

    main()
