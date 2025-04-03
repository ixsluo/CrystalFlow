# Prepare dataset from a metadata csv
#
# A example `metadata.csv` file should be like
#
# material_id,<other fields, e.g. pressure>
# <path to file>,<other fields value>
#
# And will output to
# |- output-dir
# |  |- train.csv
# |  |- val.csv
# |  |- test.csv
#
# The 'cif' field is read from file name in `material_id`
#
# Execute:
# python ../../scripts/prepare_dataset.py -m metadata.csv --split 8 1 1 -o example_dataset

import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core.structure import Structure


def spds(df, split):
    # 采样出train
    df_train = df.sample(frac=split[0])
    # 分离出val与test
    df_val   = df[~df.index.isin(df_train.index)]
    # 采样出test
    df_test  = df_val.sample(frac=split[2] / (split[1] + split[2]))
    # 分离出test得到val
    df_val   = df_val[~df_val.index.isin(df_test.index)]
    # 可选：df.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_val, df_test


def main(args):
    split = [i/sum(args.split) for i in args.split]
    df = pd.read_csv(args.meta)
    if len(df) < 10:
        raise RuntimeError(f"At least 10 data is required, but only got {len(df)}")
    cif_list = [Structure.from_file(fname).to_file(fmt="cif") for fname in df.material_id]
    df["cif"] = cif_list
    df_train, df_val, df_test = spds(df, split)

    target_dir = Path(args.outdir)
    target_dir.mkdir(exist_ok=True, parents=True)
    df_train.to_csv(target_dir.joinpath("train.csv"), index=False)
    df_val.to_csv(target_dir.joinpath("val.csv"), index=False)
    df_test.to_csv(target_dir.joinpath("test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meta', required=True, help="CSV file containing meta information.")
    parser.add_argument('-s', '--split', type=float, nargs=3, default=[8, 1, 1], help="Train-val-test split, default 8:1:1")
    parser.add_argument('-o', '--outdir', required=True, help="Target output directory.")
    args = parser.parse_args()
    main(args)
