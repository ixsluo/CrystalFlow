import argparse
import warnings
import json
from pathlib import Path
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.vasp import Vasprun
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from joblib import Parallel, delayed
from tqdm import tqdm


def load_alex_raw_convex_hull(convex_data_file) -> list:
    with open(convex_data_file, "r") as f:
        raw_convex_data = json.load(f)["entries"]
    return raw_convex_data


def find_phase_diagram(raw_convex_data, compute_entry: ComputedEntry | str) -> float:
    if compute_entry is None:
        return np.nan
    target_elements = set(map(lambda element: element.symbol, compute_entry.composition.element_composition.elements))
    if len(target_elements) >8:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    convex = [ComputedEntry.from_dict(convex_data) for convex_data in raw_convex_data if set(convex_data['data']['elements']).issubset(target_elements)]
    convex = [PDEntry(entry.composition, entry.energy) for entry in convex]
    phasediagram = PhaseDiagram(convex)
    ehull = phasediagram.get_e_above_hull(compute_entry, allow_negative=True)
    return ehull


def read_vasp_xml(vaspxml) -> tuple[str, ComputedEntry]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            vasprun = Vasprun(vaspxml)
        except Exception as e:
            print("Optimize error:", vaspxml, "ignored.")
            return (vaspxml, None)
        compute_entry = vasprun.get_computed_entry(entry_id=str(vaspxml))
    MaterialsProject2020Compatibility().process_entry(compute_entry)
    return (vaspxml, compute_entry)


def main(args):
    vaspxml_list = list(Path(args.batchrundir).rglob("vasprun.xml"))
    print("Reading vasp xml...")
    compute_entries = Parallel(args.njobs, backend="multiprocessing")(delayed(read_vasp_xml)(vaspxml) for vaspxml in tqdm(vaspxml_list[:]))
    # print(max(len(entry.composition.element_composition.elements) for vaspxml, entry in compute_entries if entry is not None))
    num_species = dict(Counter(len(entry.composition.element_composition.elements) for vaspxml, entry in compute_entries if entry is not None))
    num_species = {k: num_species[k] for k in sorted(num_species)}
    print("num species:", num_species)
    # assert 1 == 2
    print(f"Loading convex hull from {args.hull} ...")
    raw_convex_data = list(chain(*[(json.load(open(fhull, 'r'))['entries']) for fhull in args.hull]))
    print("Computing ehull...")
    # ehull_list = Parallel(args.njobs, backend="multiprocessing")(delayed(find_phase_diagram)(raw_convex_data, entry) for vaspxml, entry in tqdm(compute_entries))
    ehull_list = [find_phase_diagram(raw_convex_data, entry) for vaspxml, entry in tqdm(compute_entries)]
    name_list = [vaspxml for vaspxml, entry in compute_entries]
    formula_list = [entry.formula.replace(' ', '') if entry is not None else 'skipped' for vaspxml, entry in compute_entries]
    df = pd.DataFrame({'name': name_list, 'formula': formula_list, 'ehull': ehull_list})
    df = df.sort_values('ehull')
    df.to_csv(Path(args.batchrundir).joinpath("ehull.csv"), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('batchrundir')
    parser.add_argument('-j', '--njobs', type=int, default=1, help="default 1")
    parser.add_argument('--hull', nargs="+", help="Paths to json hull entries.")
    args = parser.parse_args()
    main(args)


# python ~/DiffCSP/scripts/readout_opt.py vasp.opt --hull ~/Data/alexandria/Alex2024/PBE3D/convex_hull_pbe.json ~/Data/MP_hull/250211/MP_PBE_stable_docs.json -j 32
