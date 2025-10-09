# SN-tSNE: Semi-Non-Negative t-SNE

**SN-tSNE** is a novel variant of *t-distributed Stochastic Neighbor Embedding (t-SNE)* that introduces **semi-non-negativity constraints** and a **multiplicative update rule (MUR)** to improve convergence stability, embedding interpretability, and cluster compactness.  
By enforcing non-negative coordinates in the low-dimensional space, SN-tSNE enhances geometric clarity while preserving the local-to-global structure balance of classic t-SNE.

---

## Published Paper
If you are interested in the detailed theoretical formulation and experimental validation, read our paper in **Knowledge-Based Systems (Elsevier)**:  
👉 **[Exploring Non-Negativity for Improved Manifold Embedding: Application to t-SNE](https://doi.org/10.1016/j.knosys.2025.114547)**

You can also check out the [C implementation (coming soon)]().

---

## ⚙️ Requirements
Ensure your environment uses **Python 3.9+** and install the following dependencies:

| Package | Version |
|----------|----------|
| numpy | 1.24.3 |
| scipy | 1.15.3 |
| pandas | 1.5.3 |
| scikit-learn | 1.6.1 |
| seaborn | 0.13.2 |
  
## Run:
To run SN-tSNE on your chosen dataset:
```
python Test_SN_tSNE.py \
  --dataset digits \
  --data-root Datasets \
  --plot-out "Plots/SN-tSNE_DIGITS.png"
```
You can replace digits with other datasets such as isolet, coil20, fashion_mnist, etc.
Results and plots will be automatically saved in the specified output path.

## Visual Overview

Below is some qualitative results.

![SN-tSNE Interpretibility1](https://github.com/M-Allaoui/SN-tSNE/blob/95928f61200f91ae956a0b150e3528c22592a500/MLP/Olivetti%20sample%201.png)
![SN-tSNE Interpretibility2](https://github.com/M-Allaoui/SN-tSNE/blob/95928f61200f91ae956a0b150e3528c22592a500/MLP/Olivetti%20sample%202.png)
![SN-tSNE Interpretibility3](https://github.com/M-Allaoui/SN-tSNE/blob/95928f61200f91ae956a0b150e3528c22592a500/MLP/Olivetti%20sample%203.png)
![SN-tSNE Interpretibility4](https://github.com/M-Allaoui/SN-tSNE/blob/95928f61200f91ae956a0b150e3528c22592a500/MLP/Olivetti%20sample%204.png)
![SN-tSNE Interpretibility5](https://github.com/M-Allaoui/SN-tSNE/blob/95928f61200f91ae956a0b150e3528c22592a500/MLP/Olivetti%20sample%205.png)

*Figure: Visualization of Olivetti face samples reconstructed from NMF, t-SNE, and SN-tSNE embeddings.*

## Citation:
If you find SN-tSNE useful in your research, please cite:

```
@article{allaoui2025exploring,
  title={Exploring Non-Negativity for Improved Manifold Embedding: Application to t-SNE},
  author={Allaoui, Mebarka and Hedjam, Rachid and Bouanane, Khadra and Allili, Mohand Sa{\"\i}d and Kherfi, Mohammed Lamine and Belhaouari, Samir Brahim},
  journal={Knowledge-Based Systems},
  pages={114547},
  year={2025},
  publisher={Elsevier}
}
```
