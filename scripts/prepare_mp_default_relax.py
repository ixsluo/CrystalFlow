import warnings
from pathlib import Path

import click

from joblib import Parallel, delayed
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def prepare_task(structure, relax_path, vaspargs={}):
    user_incar_settings = {}
    user_incar_settings.update(vaspargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mp_set = MPRelaxSet(
                structure,
                reduce_structure="niggli",
                user_incar_settings=user_incar_settings,
                user_potcar_settings={"Po": "Po_d"},
            )
            vasp = VaspInput(
                incar=mp_set.incar,
                kpoints=mp_set.kpoints,
                poscar=mp_set.poscar,
                potcar=mp_set.potcar,
            )
        except Exception as e:
            return
        relax_path.mkdir(exist_ok=True, parents=True)
        vasp.write_input(relax_path)


def wrapped_prepare_task(indir, sf, vaspargs):
    runtype = ".opt"
    relax_path = indir.with_name(f"{indir.name}{runtype}").joinpath(sf.stem)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(sf)
    try:
        prepare_task(structure, relax_path, vaspargs)
    except Exception as e:
        print(relax_path)
        raise e


@click.command
@click.argument("indirlist", nargs=-1)
@click.option("-j", "--njobs", default=-1, type=int)
@click.option("-p", "--pstress", default=0, help="PSTRESS (kbar), default 0")
def prepare_batch(indirlist, njobs, pstress):
    vaspargs = {"pstress": pstress}
    for indir in indirlist:
        indir = Path(indir)
        flist = list(indir.glob("*.vasp"))
        Parallel(njobs, backend="multiprocessing")(
            delayed(wrapped_prepare_task)(indir, sf, vaspargs)
            for sf in tqdm(flist, ncols=120)
        )


if __name__ == '__main__':
    prepare_batch()