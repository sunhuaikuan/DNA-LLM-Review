
# DNA Large Language Review

This repository contains code for reproducing the results in the paper "DNA Large Language Review" [Yuanhan et al. (2024)](https://arxiv.org/xxxx).


## Getting started in this repository


### Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.The repository  structure should look like

```bash
datasets
    |-- task01-perplexity/
        |-- gnomad.v4.1.exon.txt.gz
        |-- gnomad.v4.1.intergenic.txt.gz
        |-- gnomad.v4.1.proximity.txt.gz
    |-- task02-clustering/
        |-- Homo_sapiens.GRCh38.109.txt.gz
    |-- task03-homo-spiens/
        |-- Homo_sapiens.GRCh38.109.txt.gz
    |-- task04-pathogenecity/
        |-- clinvar_20240805.missense_matched.txt.gz
        |-- clinvar_20240805.noncoding.txt.gz
    |-- task05-methylation/
        |-- GSM6637962_CpG_coverage20_GRCh38.bed.gz
code
    |-- task01-perplexity.ipynb
    |-- task02-clustering.ipynb
    |-- task03-homo-spiens.ipynb
    |-- task04-pathogenecity.ipynb
    |-- task05-methylation.ipynb
    |-- utility.py
env
    |-- caduceus-env.yml
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
We would like to thank xxx for yyy.

