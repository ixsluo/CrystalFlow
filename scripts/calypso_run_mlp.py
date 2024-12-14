#!/usr/bin/env python
import argparse
import io
import sys
import time
import pickle
import warnings
import contextlib
from ast import literal_eval
from pathlib import Path
from enum import Enum

import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import LBFGS, BFGS, FIRE

try:
    from ase.filters import UnitCellFilter
except:
    from ase.constraints import UnitCellFilter


MODEL = Enum("model", ["chgnet@0.3.0", "dpa2", "m3gnet", "mace"])
OPTALGO = Enum("algo", ["lbfgs", "bfgs", "fire"])


def get_calculator_chgnet030(model_args: dict, calculator_args: dict) -> Calculator:
    """Get CHGNet calculator 0.2.0, 0.3.0

    Raises
    ------
    ModuleNotFoundError
        If chgnet python package not installed.

    References
    ----------
    https://github.com/CederGroupHub/chgnet/blob/main/chgnet/model/dynamics.py#L51
    """
    try:
        from chgnet.model import CHGNet
        from chgnet.model import CHGNetCalculator
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("You select CHGNet but chgnet package not installed.") from e
    model_args = {
        **{},
        **model_args,
        **{
            'use_device': None,
            'check_cuda_mem': False,
        }
    }
    calculator_args = {
        **{
            'on_isolated_atoms': 'warn',
        },
        **calculator_args,
        **{
            'use_device': None,
            'check_cuda_mem': False,
            'stress_weight': 0.006241509125883258,
        },
    }
    chgnet = CHGNet.load(model_name="0.3.0", **model_args)
    calc = CHGNetCalculator(model=chgnet, **calculator_args)
    return calc


def get_calculator_dpa2(model_args: dict, calculator_args: dict) -> Calculator:
    """Get DPA2 calculator

    Raises
    ------
    ModuleNotFoundError
        If deepmd python package not installed.
    ValueError
        If model pt path not specified, eg ``--calculator-args="{'model': '/path/to/pb/file'}"``

    References
    ----------
    https://github.com/deepmodeling/deepmd-kit/blob/r2/deepmd/calculator.py#L34
    """
    try:
        from deepmd.calculator import DP
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("You select DP but deepmd package not installed.") from e
    calculator_args = {
        **{},  # overwrittenable
        **model_args,
        **calculator_args,
        **{},  # non-overwrittenable
    }
    if "model" not in calculator_args:
        raise ValueError("""You must specify --calculator-args="{'model': '/path/to/pb/file'}" when using DP.""")
    calc = DP(**calculator_args)
    return calc


def get_calculator_m3gnet(model_args: dict, calculator_args: dict) -> Calculator:
    """Get M3GNet calculator

    Raises
    ------
    ModuleNotFoundError
        If matgl python package not installed.

    References
    ----------
    https://github.com/materialsvirtuallab/matgl/blob/main/src/matgl/ext/ase.py#L124
    """
    try:
        import matgl
        from matgl.ext.ase import PESCalculator
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("You select m3gnet but matgl package not installed.") from e
    # CHGNet-MPtrj-2023.12.1-PES-2.7M, CHGNet-MPtrj-2024.2.13-PES-11M, M3GNet-MP-2018.6.1-Eform, M3GNet-MP-2021.2.8-DIRECT-PES,
    # M3GNet-MP-2021.2.8-PES, MEGNet-MP-2018.6.1-Eform, MEGNet-MP-2019.4.1-BandGap-mfi,
    model_args = {
        **{'path': "M3GNet-MP-2021.2.8-PES"},  # overwrittenable
        **model_args,
        **{},  # non-overwrittenable
    }
    m3gnet = matgl.load_model(**model_args)
    # stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa
    calculator_args = {
        **{},  # overwrittenable
        **calculator_args,
        **{
            "stress_weight": 1 / 160.21766208,
        },  # non-overwrittenable
    }
    calc = PESCalculator(potential=m3gnet, **calculator_args)
    return calc


def get_calculator_mace(model_args: dict, calculator_args: dict) -> Calculator:
    """Get MACE calculator

    Raises
    ------
    ModuleNotFoundError
        If mace python package not installed.

    References
    ----------
    https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py#L18
    """
    try:
        from mace.calculators import MACECalculator
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("You select mace but mace package not installed.") from e
    # [MACE, DipoleMACE, EnergyDipoleMACE]
    calculator_args = {
        **{  # overwrittenable
            "device": "cuda",
        },
        **model_args,
        **calculator_args,
        **{
            "energy_units_to_ev": 1.0,
            "length_units_to_A": 1.0,
            "model_type": "MACE",
        },  # non-overwrittenable
    }
    calc = MACECalculator(model_path="", **calculator_args)
    return calc


def get_calculator_mace_mp(model_args: dict, calculator_args: dict) -> Calculator:
    """Get MACE(mace_mp) calculator

    Raises
    ------
    ModuleNotFoundError
        If mace python package not installed.

    References
    ----------
    https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py#L18
    """
    try:
        from mace.calculators import mace_mp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("You select mace_mp but mace package not installed.") from e
    calculator_args = {
        **{
            "model": "medium",
            "device": "cuda",
        },  # overwrittenable
        **model_args,
        **calculator_args,
        **{},  # non-overwrittenable
    }
    calc = mace_mp(**calculator_args)
    return calc


def get_calculator(modelname: str, model_args: str = "{}", calculator_args: str = "{}") -> Calculator:
    modelname = modelname.lower()
    model_args = literal_eval(model_args)
    calculator_args = literal_eval(calculator_args)
    if modelname == "chgnet@0.3.0":
        calc = get_calculator_chgnet030(model_args, calculator_args)
    elif modelname == "dpa2":
        calc = get_calculator_dpa2(model_args, calculator_args)
    elif modelname == "m3gnet":
        calc = get_calculator_m3gnet(model_args, calculator_args)
    elif modelname == "mace":
        calc = get_calculator_mace(model_args, calculator_args)
    elif modelname == "mace_mp":
        calc = get_calculator_mace_mp(model_args, calculator_args)
    else:
        raise ValueError(f"Unknown model name: {modelname}.")
    return modelname, calc


def get_optimizer_algo(optimizername: str):
    if optimizername.lower() == "lbfgs":
        return LBFGS
    elif optimizername.lower() == "bfgs":
        return BFGS
    elif optimizername.lower() == "fire":
        return FIRE
    else:
        raise ValueError(f"Unknown optimizer name: {optimizername}")


def run_opt(
    cif: str | Path,
    modelname: str,
    calculator: Calculator,
    opt_algo,
    pstress: float,  # GPa
    fmax: float,
    optsteps: int,
    traj_suffix: str,
    opt_suffix: str,
    optinfo_suffix: str,
    verbose=True,
):
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        print(f"Start optimizing {cif}...")
        start = time.time()

        data = {}
        cif = Path(cif)
        # atoms = read(cif, format="cif")
        atoms = read(cif)

        dis_mtx = atoms.get_all_distances(mic=True)
        row, col = np.diag_indices_from(dis_mtx)
        dis_mtx[row, col] = np.min(atoms.cell.lengths())
        min_dis = np.nanmin(dis_mtx)
        volume = float(atoms.get_volume())
        force = np.full_like(atoms.get_scaled_positions(), np.nan).tolist()
        energy = np.nan
        stress = [np.nan] * 6
        enthalpy = np.nan

        if min_dis > 0.6:
            atoms.calc = calculator
            aim_stress = 1.0 * pstress * 0.01 * 0.6242
            ucf = UnitCellFilter(atoms, scalar_pressure=aim_stress)

            traj_file = str(cif.with_suffix(traj_suffix))
            opt = opt_algo(ucf, trajectory=traj_file)

            try:
                opt.run(
                    fmax=fmax,
                    steps=optsteps,
                )
            except IndexError as e:
                mlp_opt_state = False
                warnings.warn(f"Optimization failed. Single atom with chgnet may raise this error: {e}")
            except Exception as e:
                mlp_opt_state = False
                warnings.warn(f"Optimization failed. Raise from error: {e}")
            else:
                volume = float(atoms.get_volume())
                force = np.array(atoms.get_forces()).tolist()
                energy = float(atoms.get_potential_energy())
                stress = np.array(atoms.get_stress(voigt=True) * 160.21766028).tolist()  # eV/A^3 to GPa
                enthalpy = energy + pstress * volume * 0.01 * 0.6242
                mlp_opt_state = True
        else:
            mlp_opt_state = False
            warnings.warn(f"The minimum distance of two atoms is {min_dis}, too close.")

        symbols = atoms.get_chemical_symbols()
        lattice = np.array(atoms.get_cell()).tolist()
        position = np.array(atoms.get_positions()).tolist()

        stop = time.time()
        cost = stop - start

        data["lattice"] = lattice
        data["symbols"] = symbols
        data["cartpos"] = position
        data["force"] = force
        data["volume"] = volume
        data["energy"] = energy
        data["stress"] = stress
        data["enthalpy"] = enthalpy
        data["pullay_stress"] = pstress
        data["opt_setting"] = {}
        data["opt_setting"]["optimizer"] = modelname
        data["opt_setting"]["fmax"] = fmax
        data["opt_setting"]["mlp_opt_state"] = mlp_opt_state
        data["opt_setting"]["time_consuming"] = cost
        data["opt_setting"]["converge"] = True if np.max(np.abs(force)) <= fmax else False

        with open(cif.with_suffix(optinfo_suffix), "wb") as f:
            pickle.dump(data, f)

        opt_atoms = Atoms(symbols=data.pop('symbols'), positions=data.pop('cartpos'), cell=data.pop('lattice'))
        sg_calc = SinglePointCalculator(
            opt_atoms, energy=data.pop('energy'), forces=data.pop('force'), stress=data.pop("stress")
        )
        opt_atoms.calc = sg_calc
        opt_atoms.info = data
        write(cif.with_suffix(opt_suffix), opt_atoms)

        print(f"opt done! cost {cost} seconds.")


def main(args):
    modelname, calculator = get_calculator(args.model, args.model_args, args.calculator_args)
    optimizer_algo = get_optimizer_algo(args.algo)

    for cif in args.cif:
        run_opt(
            cif=cif,
            modelname=modelname,
            calculator=calculator,
            opt_algo=optimizer_algo,
            pstress=args.pressure,
            fmax=args.fmax,
            optsteps=args.optsteps,
            traj_suffix=args.traj_suffix,
            opt_suffix=args.opt_suffix,
            optinfo_suffix=args.optinfo_suffix,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cif", nargs="+", help="Structure cif file to be optimized.")
    parser.add_argument("-m", "--model", choices=[m.name for m in MODEL], default=MODEL(1).name, help="Machine learning potential to use.")
    parser.add_argument("--model-args", default="{}", help="Other model arguments as dict string.")
    parser.add_argument("--calculator-args", default="{}", help="Other calculator arguments as dict string.")
    parser.add_argument("-a", "--algo", choices=[c.name for c in OPTALGO], default=OPTALGO(1).name, help="Optimization algorithem.")
    parser.add_argument("-P", "--pressure", type=float, default=50.0, help="External pressure, in GPa.")
    parser.add_argument("-F", "--fmax", type=float, default=0.1, help="Max force for optimizations, in eV/A.")
    parser.add_argument("-N", "--optsteps", type=int, default=200, help="Number of optimizer steps to be run.")
    parser.add_argument("--traj-suffix", default=".opt.traj", help="Trajectory filename suffix.")
    parser.add_argument("--opt-suffix", default=".opt.cif", help="Optimized structure filename suffix.")
    parser.add_argument("--optinfo-suffix", default=".opt.pkl", help="Settings and info of optimized structure pickle filename suffix.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    # fmt: on

    args = parser.parse_args()

    main(args)
