# SHARK: Shapley Reweighted k-means

[![Paper](https://img.shields.io/badge/arXiv-2508.07952-b31b1b.svg)](https://arxiv.org/abs/2508.07952)

This repository contains the official implementation of **SHARK (Shapley Reweighted k-means)**,  
a feature-weighted clustering algorithm introduced in:

> **Shapley-Inspired Feature Weighting in k-means with No Additional Hyperparameters**  
> Richard J. Fawley, Renato Cordeiro de Amorim  
> University of Essex  
> [[arXiv:2508.07952]](https://arxiv.org/abs/2508.07952)

---

## 📌 Overview

Classical k-means assumes all features contribute equally to the clustering structure,  
which often fails in high-dimensional or noisy datasets.  

**SHARK** addresses this by:
- Using **Shapley values** from cooperative game theory to quantify feature relevance.
- Iteratively **reweighting features** inversely proportional to their Shapley contributions.
- Requiring **no additional hyperparameters** beyond those in k-means.

This results in more **robust and accurate clustering**, especially in noisy settings,  
while retaining the simplicity and efficiency of k-means.

---

## 🚀 Key Features

- ✅ No additional hyperparameters (only `k` is required, as in k-means).  
- ✅ Closed-form computation of Shapley values (polynomial time).  
- ✅ Robust to irrelevant/noisy features.  
- ✅ Outperforms or matches **k-means++**, **FWSA**, and **LW-k-means** on synthetic and real-world datasets.  
- ✅ Theoretically grounded: objective is guaranteed to be no worse than comparable k-means.  

---

## 📖 Algorithm

SHARK modifies the k-means iterative procedure by:  

1. Initialising with equal feature weights.  
2. Computing per-feature Shapley values of dispersion.  
3. Updating feature weights as the **inverse Shapley value**, normalised to sum to 1.  
4. Reassigning points and updating centroids as in k-means.  
5. Repeating until convergence.  

This minimises the **harmonic mean of feature-wise dispersions**,  
as opposed to the arithmetic mean minimised by k-means.  

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/rickfawley/shark.git
cd shark

## Generating Synthetic Datasets

To reproduce the synthetic benchmarks described in the [SHARK paper](https://arxiv.org/abs/2508.07952), 
we provide a helper script:

```bash
python scripts/generate_datasets.py --out ./data --n-repeats 50 --normalize range --seed 42

