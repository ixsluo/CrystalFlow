import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pyxtal import pyxtal
from scipy.stats import wasserstein_distance
from tqdm import tqdm
# from joblib import Parallel, delayed

sys.path.append('.')

from eval_utils import (
    CompScaler,
    compute_cov,
    get_crystals_list,
    get_fp_pdist,
    load_config,
    load_data,
    prop_model_eval,
    smact_validity,
    structure_validity,
)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict, compute_valid=True, compute_fp=True, ignore_smact=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()

        self.ignore_smact = ignore_smact
        if compute_valid:
            self.get_validity()
        else:
            self.valid = self.comp_valid = self.struct_valid = True

        if compute_fp:
            self.get_fingerprints()
        else:
            self.comp_fp = self.struct_fp = None


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'
        if not hasattr(self, "constructed"):
            raise ValueError("XXX")

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        if len(self.elems) >= 8:
            self.comp_valid = False
        else:
            self.comp_valid = smact_validity(self.elems, self.comps) if not self.ignore_smact else True
        if self.constructed:
            if _is_odd(self.structure):
                self.struct_valid = False
            else:
                self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                site_fps = [CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def _is_odd(structure: Structure):
    lengths = np.array(structure.lattice.abc)
    angles = np.array(structure.lattice.angles)
    if any(angles < 10) or any(angles > 170):
        return True
    elif any(lengths / np.power(len(structure), 1 / 3) > 20):
        return True
    elif any(lengths / np.power(len(structure), 1 / 3) < 0.1):
        return True
    elif not (0.1 < (structure.volume / len(structure)) < 100):
        return True
    return False


MAX_RMSD = 0.5


def get_rms_dist(pred: Crystal, gt, is_valid, matcher, assess_all=False, norm=True):

    def av_lat(l1: Lattice, l2: Lattice):
        params = (np.array(l1.parameters) + np.array(l2.parameters)) / 2
        return Lattice.from_parameters(*params)

    if not pred.constructed:
        rms_dist = None
    elif _is_odd(pred.structure):
        rms_dist = None
    if not is_valid:
        rms_dist = None

    try:
        rms_dist = matcher.get_rms_dist(pred.structure, gt.structure)
        rms_dist = rms_dist[0]
    except Exception:
        rms_dist = None

    if (rms_dist is None) and assess_all:
        rms_dist = MAX_RMSD

    if norm:
        return rms_dist
    else:
        avg_l = av_lat(pred.structure.lattice, gt.structure.lattice)
        normalization = (len(pred.structure) / avg_l.volume) ** (1 / 3)
        return rms_dist / normalization


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3, njobs=1):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.njobs = njobs

    def get_match_rate_and_rms(self):
        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        rms_dists = p_map(
            partial(get_rms_dist, matcher=self.matcher),
            self.preds,
            self.gts,
            validity,
            num_cpus=self.njobs,
            ncols=79,
        )
        self.rms_dists = rms_dists

        # rms_dists = []
        # for i in tqdm(range(len(self.preds)), ncols=79):
        #     rms_dists.append(process_one(
        #         self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)

        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            return get_rms_dist(pred, gt, is_valid, self.matcher)
            # if not is_valid:
            #     return None
            # try:
            #     rms_dist = self.matcher.get_rms_dist(
            #         pred.structure, gt.structure)
            #     rms_dist = None if rms_dist is None else rms_dist[0]
            #     return rms_dist
            # except Exception:
            #     return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist
        }

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics

class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 

def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'opt' in args.tasks:
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)

    elif 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = -2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'], num_cpus=args.njobs)
        else:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)


    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                if args.multi_idx is not None and i != args.multi_idx:
                    continue
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i], num_cpus=args.njobs))

        if args.multi_eval:
            rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
            # rec_evaluator = RecEvalBatch(pred_crys, gt_crys, stol=0.3, angle_tol=5, ltol=0.2)
        else:
            rec_evaluator = RecEval(pred_crys, gt_crys)

        recon_metrics = rec_evaluator.get_metrics()

        if hasattr(rec_evaluator, "all_rms_dis"):
            all_metrics["all_rms_dis"] = rec_evaluator.all_rms_dis.tolist()

        all_metrics.update(recon_metrics)



    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    if args.multi_idx is not None:
        metrics_out_file = str(Path(metrics_out_file).stem + f"_{args.multi_idx}.json")
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--multi_idx', type=int, default=None, help="index for multi_eval (special case)")
    parser.add_argument('-j', '--njobs', default=32, type=int)
    args = parser.parse_args()
    main(args)
