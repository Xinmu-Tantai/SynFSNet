

# **AFSIFormer: Adaptive Frequency–Spatial Interaction Attention Mechanism for Aerial Image Semantic Segmentation**

**SynFSNet** is a state-of-the-art semantic segmentation network specifically designed for **high-resolution aerial imagery**, integrating frequency-domain reasoning with spatial feature modeling to achieve **precise and efficient segmentation** of complex scenes.

📄 [**Paper Link (IEEE TGRS2025)**](https://ieeexplore.ieee.org/abstract/document/11126528/)

---

## 🔑 **Core Innovations**

<img width="2259" height="1247" alt="image" src="https://github.com/user-attachments/assets/6d1c9473-cc9d-4210-bc1a-4bc8944105e8" />


### 🧠 **AFSIAttention**

Enables **fine-grained interaction between frequency and spatial domains** via head-specific frequency projections and adaptive weighting. This allows the network to capture **subtle structural and textural details**, crucial for aerial imagery analysis.

### 🌐 **AFSIFormer Block**

Implements **progressive learning** with **boundary-aware directional attention** and **local window attention**, achieving **global-local feature synergy** while maintaining computational efficiency.

### 🔄 **Block-Level Residual Coupling**

A novel architecture that **continuously enhances global and local feature representations**, ensuring stable information flow and reinforcing semantic consistency across layers.

### 🧩 **BiSEM Module (Bidirectional Semantic Enhancement Module)**

Bridges the semantic gap between **low-level and high-level features**, significantly improving the perception of **fine-grained structures** such as roads, buildings, and other small-scale objects.

### ⚡ **Lightweight & Efficient**

Balances **high segmentation accuracy, real-time inference speed, and strong generalization**, making SynFSNet highly suitable for **real-world aerial image applications**.

---

## 📑 **Citation**

```bibtex
@ARTICLE{11126528,
  author={Hui, Jie and Mi, Wenyu and Wang, Jianji and Cao, Yuanyang and Zhou, Ziyi and Zheng, Nanning},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={AFSIFormer: Adaptive Frequency–Spatial Interaction Attention Mechanism for Aerial Image Semantic Segmentation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-19},
  keywords={Frequency-domain analysis;Feature extraction;Semantic segmentation;Computational modeling;Transformers;Data mining;Computer architecture;Semantics;Discrete cosine transforms;Couplings;Aerial image segmentation;attention mechanism;frequency–spatial interaction;remote sensing;residual coupling architecture},
  doi={10.1109/TGRS.2025.3599214}}
}
```

---





