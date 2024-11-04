
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

## Datasets and Tasks

### Perplexity

gnomad.v4.1.exon.txt.gz includes extremely rare (private) variants in exons: missense, synonymous, splice-donor/acceptor, gain or loss of variants.

gnomad.v4.1.proximity.txt.gz includes variants in proximity regions of exons: intronic, 3’ or 5’ UTR, splice-region, and exon non-coding variants.

gnomad.v4.1.intergenic.txt.gz includes intergenic, upstream/downstream, and regulatory-region variants.

### Genomic region clustering

Homo_sapiens.GRCh38.109.txt.gz are Short genomic regions (<1000 ish base pairs, small RNAs >40 bp, ncRNA > 110 bp, all others > 150 bp) extracted from ensemble human annotations includeing  first exon, first intron, first 3’ UTR, first 5’ UTR, ncRNA, pseudogene, and small RNAs.

### Supervised pathogenicity predictions

We collected 19,333 overlapping variants from ClinVar and PrimateAI datasets.  1.4% are potentially pathogenic.

<table>
<tr>
    <td>Pathogenic</td><td>Likely_Pathogenic</td><td>Pathogenic_VUS</td><td>VUS_Pathogenic</td><td>VUS</td><td>VUS_Benign</td><td>Benign_VUS</td><td>Likely_Benign</td><td>Benign</td>
</tr>
<tr>
    <td>114	</td><td>101</td><td>22</td><td>36</td><td>15421</td><td>1980</td><td>326</td><td>949</td><td>384</td>
</tr>
<table>

We consider interpreting missense variants and non-coding variants (including intergenic region, intron, non-coding, 3’ and 5’ UTR). The missense dataset contains 8.35% pathogenic.The noncoding dataset contains 1.34% pathogenic.


### Supervised methylation percentage predictions

To have a meaning methylation percentage, methylation dataset includes sites with depth > 20. The target is to predict the percentage. We use a flank region of 128  around the site and extract the averaged embedding for the two base pairs.

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

