# Crystal Structure Prediction by Joint Equivariant Diffusion (NeurIPS 2023)

Implementation codes for Crystal Structure Prediction by Joint Equivariant Diffusion (DiffCSP). 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jiaor17/DiffCSP/blob/main/LICENSE)   [**[Paper]**](https://arxiv.org/abs/2309.04475)

![Overview](fig/overview.png "Overview")

![Demo](fig/demo.gif "Demo")

### Dependencies and Setup

Note: updated `python==3.11` and `pytorch>=2.0.0` and `hydra>=1.3`

```bash
# sudo apt-get install gfortran libfftw3-dev pkg-config
conda create -n diffcsp python=3.11.9
conda activate diffcsp
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install lightning==2.3.2
pip install hydra-core omegaconf python-dotenv wandb rich
pip install p_tqdm pymatgen pyxtal smact matminer einops chemparse torchdyn
pip install -e .
mkdir log
```

Rename the `.env.template` file into `.env` and specify the following variables.

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WANDB_DIR: the absolute path to save wabdb outputs
```

example:
```bash
# vi .env
export PROJECT_ROOT="/home/<YOURHOME>/DiffCSP"
export   HYDRA_JOBS="/home/<YOURHOME>/DiffCSP/hydra"
export    WANDB_DIR="/home/<YOURHOME>/DiffCSP/log"
```

### Training

For the CSP task

```bash
python diffcsp/run.py data=<dataset> expname=<expname>

# example:
# python diffcsp/run.py data=mp_20 logging.wandb.group=mp_20 expname=origin
```

For the Ab Initio Generation task

```bash
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24, and the ``<expname>`` tag can be an arbitrary name to identify each experiment. Pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/11WOc9lTZN4hkIY7SKLCIrbsTMGy9TsoW?usp=sharing).

### Evaluation

#### Stable structure prediction 

One sample 

```bash
python scripts/evaluate.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv 

# example:
# python ~/DiffCSP/scripts/evaluate.py --model_path `pwd` --dataset mp_20
# python ~/DiffCSP/scripts/compute_metrics.py --root_path `pwd` --tasks csp --gt_file ~/DiffCSP/data/mp_20/test.csv
```

Multiple samples

```bash
python scripts/evaluate.py --model_path <model_path> --dataset <dataset> --num_evals 20
python scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --multi_eval
```

#### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics.py --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```


#### Sample from arbitrary composition

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --formula <formula> --num_evals <num_evals>
```

#### Property Optimization

```
# train a time-dependent energy prediction model 
python diffcsp/run.py data=<dataset> model=energy expname=<expname> data.datamodule.batch_size.test=100

# Optimization
python scripts/optimization.py --model_path <energy_model_path> --uncond_path <model_path>

# Evaluation
python scripts/compute_metrics.py --root_path <energy_model_path> --tasks opt
```

### Acknowledgments

The main framework of this codebase is build upon [CDVAE](https://github.com/txie-93/cdvae). For the datasets, Perov-5, Carbon-24 and MP-20 are from [CDVAE](https://github.com/txie-93/cdvae), and MPTS-52 is collected from its original [codebase](https://github.com/sparks-baird/mp-time-split).

### Citation

Please consider citing our work if you find it helpful:
```
@article{jiao2023crystal,
  title={Crystal structure prediction by joint equivariant diffusion},
  author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023}
}
```

### Contact

If you have any questions, feel free to reach us at:

Rui Jiao: [jiaor21@mails.tsinghua.edu.cn](mailto:jiaor21@mails.tsinghua.edu.cn)
