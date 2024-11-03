
# DNA Large Language Review

This repository contains code for reproducing the results in the paper "DNA Large Language Review" [Huaikuan et al. (2024)](https://arxiv.org/xxxx).


## Getting started in this repository

### Prerequisite 

Download the entire human genome fasta file [hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz).

### Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper. The repository  structure should look like

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
    |-- task04-pathogenicity/
        |-- clinvar_20240805.missense_matched.txt.gz
        |-- clinvar_20240805.noncoding.txt.gz
    |-- task05-methylation/
        |-- GSM6637962_CpG_coverage20_GRCh38.bed.gz
code
    |-- task01-perplexity.ipynb
    |-- task02-clustering.ipynb
    |-- task03-homo-spiens.ipynb
    |-- task04-pathogenicity.ipynb
    |-- task05-methylation.ipynb
    |-- utility.py
env
    |-- nt-env.yml
    |-- gpn-env.yml
    |-- hyena-env.yml
    |-- caduceus-env.yml
```

## Tasks

### Perplexity

### Genomic region clustering

### Supervised pathogenicity predictions

### Supervised methylation percentage predictions

## Citation
<a name="citation"></a>

If you find our work useful, please cite our paper using the following:
```
@article{DNALLMREVIW2024UFBME,
  title={DNA Large lanuage Model Review},
  author={Sun, Huaikuan and Chen, Yuanhan and Fan, Xiao},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2024}
}
```

## Acknowledgements
<a name="acknowledgements"></a>
We would like to thank xxx for yyy.

