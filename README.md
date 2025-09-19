# CrystalFlow: A Flow-Based Generative Model for Crystalline Materials

Implementation codes for CrystalFlow: A Flow-Based Generative Model for Crystalline Materials.

For MindSpore implementation please refer to [here](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/crystalflow).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ixsluo/CrystalFlow/blob/main/LICENSE)   [**[Paper]**](https://arxiv.org/abs/2412.11693)

### Dependencies and Setup

Note: updated `python==3.11` and `pytorch>=2.0.0` and `hydra>=1.3`

```bash
# sudo apt-get install gfortran libfftw3-dev pkg-config
conda create -n crystalflow python=3.11.9
conda activate crystalflow
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install lightning==2.3.2 finetuning_scheduler
pip install hydra-core omegaconf python-dotenv wandb swanlab rich
pip install p_tqdm pymatgen pyxtal smact matminer einops chemparse torchdyn
pip install -e .
mkdir -p log hydra
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
export PROJECT_ROOT="/home/<YOURHOME>/CrystalFlow"
export   HYDRA_JOBS="/home/<YOURHOME>/CrystalFlow/hydra"
export    WANDB_DIR="/home/<YOURHOME>/CrystalFlow/log"
```

### Datasets

MP20 and MPTS52 datasets are provided in `dataset/mp_20` and `dataset/mpts_52`.

### Training

test passed on version `v1.0.0-alpha.1`.

Pretrained checkpoints are provided in [Releases](https://github.com/ixsluo/CrystalFlow/releases).

#### For CSP task on MP-20 dataset without conditioning

```bash
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
data=mp_20 data.train_max_epochs=3000 \
model=flow_polar \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=CSP-mp20 \
      > CSP-mp20.log 2>&1 &
```

The checkpoints and other files will be in `hydra/singlerun/CSP-mp20`.

Run time on RTX-4090 is about 24 hours.

#### For CSP task on MP-CALYPSO-60 dataset with pressure conditioning

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
data=calypso_60 data.train_max_epochs=2000 \
data.datamodule.batch_size.train=64 \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
model=flow_polar \
+model.guide_threshold=-1 \
+model.from_cubic=false \
+model.lattice_polar_sigma=0.1 \
model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=CSP-mpcalypso60-pressure \
      > CSP-mpcalypso60-pressure.log 2>&1 &
```

#### For DNG task on MP-20 without conditioning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
data=mp_20 data.train_max_epochs=3000 \
model=flow_polar_w_type \
+model.type_encoding=table \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_type=10 model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=DNG-mp20 \
      > DNG-mp20.log 2>&1 &
```

#### For DNG task on MP-20 with formation energy conditioning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
data=mp_20_chgnet data.train_max_epochs=3000 \
model=flow_polar_w_type \
+model.type_encoding=table \
+model.guide_threshold=-1 \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_type=10 model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=DNG-mp20-Eform \
      > DNG-mp20-Eform.log 2>&1 &
```


### Evaluation

#### CSP generation

One sample

```bash
python /path/to/scripts/evaluate.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --anneal_coords --anneal_slope 5 --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

Multiple samples

```bash
python /path/to/scripts/evaluate.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --num_evals 20 --anneal_coords --anneal_slope 5 --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --multi_eval --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

##### extract CSP generation

```bash
python /path/to/scripts/extract_gen.py eval_gen_<label>.pt --task eval
# will save to eval_gen_<label>.dir
```

#### DNG generation

```bash
python /path/to/scripts/generation.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

##### extract DNG generation

```bash
python /path/to/scripts/extract_gen.py eval_gen_<label>.pt --task gen
# will save to eval_gen_<label>.dir
```


#### Sample from arbitrary composition

```bash
python /path/to/scripts/sample.py --model_path <model_path> --ode-int-steps 100 --save_path <save_path> --formula <formula> --num_evals <num_evals> --anneal_coords --anneal_slope 5
# will save to the <save_path>
```

### Acknowledgments

The main framework of this codebase is build upon [DiffCSP](https://github.com/txie-93/cdvae). For the datasets, Perov-5, Carbon-24 and MP-20 are from [CDVAE](https://github.com/txie-93/cdvae), and MPTS-52 is from [DiffCSP](https://github.com/txie-93/cdvae) (originaly from [codebase](https://github.com/sparks-baird/mp-time-split)).

### Citation

Please consider citing our work if you find it helpful:
```
X. Luo et al., CrystalFlow: A Flow-Based Generative Model for Crystalline Materials, arXiv 2412.11693
```

### Contact

If you have any questions, feel free to reach us at:

Xiaoshan Luo: [luoxs21@mails.jlu.edu.cn](mailto:luoxs21@mails.jlu.edu.cn)
