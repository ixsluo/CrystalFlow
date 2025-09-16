import argparse

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher


def main(args):
    print(vars(args))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(description="Structure generation mode", dest="mode")
    cspparser = subparsers.add_parser("CSP", help="parse CSP results")
    dngparser = subparsers.add_parser("DNG", help="parse DNG results")

    cspparser.add_argument("ckpt", help="checkpoint file name")
    cspparser.add_argument("resultdir", help="result directory")
    cspparser.add_argument("-j", "--njobs", default=32, type=int)

    dngparser.add_argument("ckpt", help="checkpoint file name")
    dngparser.add_argument("resultdir", help="result directory")
    dngparser.add_argument("-j", "--njobs", default=32, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
