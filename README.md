# DNA-LLM-Review
 code for DNA Large Language Model Review



# DNA Large Language Review

This repository contains code for reproducing the results in the paper "DNA Large Language Review" [Yuanhan et al. (2024)](https://arxiv.org/xxxx).


## Getting started in this repository
<a name="getting_started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f caduceus_env.yml
```

Activate the environment.

```bash
conda activate caduceus_env
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

## Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`train.py`](./train.py) script.
We also provide sample `slurm` scripts for launching pre-training and downstream fine-tuning experiments in the [`slurm_scripts/`](./slurm_scripts) directory.

### Pretraining on Human Reference Genome
<a name="pretraining"></a>
(Data downloading instructions are copied from [HyenaDNA repo](https://github.com/HazyResearch/hyena-dna?tab=readme-ov-file#pretraining-on-human-reference-genome))

First, download the Human Reference Genome data.
It's comprised of 2 files, 1 with all the sequences (the `.fasta` file), and with the intervals we use (`.bed` file).

The file structure should look like

```
data
|-- hg38/
    |-- hg38.ml.fa
    |-- human-sequences.bed
```

Download fasta (.fa format) file (of the entire human genome) into `./data/hg38`.
~24 chromosomes in the whole genome (merged into 1 file), each chromosome is a continuous sequence, basically.
Then download the .bed file with sequence intervals (contains chromosome name, start, end, split, which then allow you to retrieve from the fasta file).
```bash
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
gunzip data/hg38/hg38.ml.fa.gz  # unzip the fasta file
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
```

Launch pretraining run using the command line

```bash
python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=1024 \
  dataset.batch_size=1024 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug=false \
  model=caduceus \
  model.config.d_model=128 \
  model.config.n_layer=4 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=true \
  optimizer.lr="8e-3" \
  train.global_batch_size=1024 \
  trainer.max_steps=10000 \
  +trainer.val_check_interval=10000 \
  wandb=null
```

or alternatively, if using a cluster that has `slurm` installed, adapt the scripts below:
```
slurm_scripts
|-- run_pretrain_caduceus.sh
|-- run_pretrain_hyena.sh
|-- run_pretrain_mamba.sh
```

and run the training as a batch job:
```bash
cd slurm_scripts
sbatch run_pretrain_caduceus.sh
```

## Citation
<a name="citation"></a>

If you find our work useful, please cite our paper using the following:
```
@article{DNALLMREVIW2024UFBME,
  title={DNA Large lanuage Model Review},
  author={Chen, Yuanhan and Sun, Huaikuan and Fan, Xiao},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2024}
}
```

## Acknowledgements
<a name="acknowledgements"></a>
We would like to thank ...

