import argparse
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout

from sym_graphs.data_io.dataset import MyGEDDataset
from sym_graphs.data_io.rpe import MyDataProcessor

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run MyGEDDataset with configurable parameters.")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=os.path.join(script_dir, "../../configs/emu_cfg/config_small.yaml"),
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(script_dir, "../../dataset/benchmarks"),
        help="Path to datasets root folder.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="aids",
        choices=[
            "aids",
            "linux",
            "mutagenicity",
            "ogbg-code2",
            "ogbg-molhiv",
            "ogbg-molpcba",
            "yeast",
        ],
        help="Name of the dataset to emulate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset mode : train, val or test",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="equal",
        choices=["equal", "unequal", "label"],
        help="Dataset type : equal, unequal or label.",
    )
    parser.add_argument(
        "--emulator",
        type=str,
        default="SV",
        choices=["SV", "MPS"],
        help="which emulator to use",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Shows emulation progress when True",
    )
    parser.add_argument(
        "--upper_lim",
        type=int,
        default=20,
        help="Don't account for data with more than this threshold.",
    )

    args = parser.parse_args()

    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        correlation_initiator = MyDataProcessor(
            args.cfg_file,
            args.upper_lim,
            filter_type="upto",
            emulator=args.emulator,
            verbose=args.verbose,
        )

        _ = MyGEDDataset(
            root=args.data_root,
            name=args.name,
            mode=args.mode,
            dataset_type=args.type,
            pre_filter=correlation_initiator.filter_out,
            pre_transform=correlation_initiator.compute_correlation,
            force_reload=True,
        )


if __name__ == "__main__":
    main()
