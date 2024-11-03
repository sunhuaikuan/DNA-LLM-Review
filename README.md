
# DNA Large Language Review

This repository contains code for reproducing the results in the paper "DNA Large Language Review" [Huaikuan et al. (2024)](https://arxiv.org/xxxx).


## Getting started in this repository


### Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.

Download fasta (.fa format) file (of the entire human genome).

The repository  structure should look like

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

