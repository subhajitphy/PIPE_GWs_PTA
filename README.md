# PIPE_GWs_PTA
**Physics-Informed Transformers + Simulation-Based Inference for Gravitational Waves in PTA Data**

This repository contains the **code, models, and datasets** for the paper:

**Transformers with Physics-informed encodings and Simulation-Based inference for robust Gravitational-Wave detection in Pulsar Timing Array data**
*AI4X – Accelerate Conference 2026*

📄 Paper: https://openreview.net/forum?id=Bpo4HgB6by
📄 PDF: [Download PDF](https://drive.google.com/file/d/14FwFw048rX4QX9m1_37sWwWtM_pN4-Ba/view?usp=sharing)

---

## 🚀 Overview

Pulsar Timing Arrays (PTAs) probe **nanohertz gravitational waves**, but traditional Bayesian inference methods are computationally expensive in high-dimensional, noisy settings.

This project introduces a **physics-informed machine learning pipeline** that:

- Uses **Transformer architectures** with physics-aware encodings
- Embeds **orbital phase evolution directly into the model**
- Combines deep learning with **Simulation-Based Inference (SBI)**
- Produces **fast, accurate posterior distributions**

➡️ Achieves **orders-of-magnitude faster inference** compared to MCMC-based methods.

---

## 🧠 Key Contributions

- **Physics-Informed Positional Encoding (PIPE)**
  Encodes orbital phase evolution into Transformer embeddings

- **External-Attention Transformer**
  Efficient long-range sequence modeling

- **Phase + SNR Prediction Network**
  Recovers latent GW phase from noisy PTA signals

- **Simulation-Based Inference (SBI)**
  - Discrete Normalizing Flows (DNF)
  - Continuous Normalizing Flows (CNF)

- **End-to-End Pipeline**
  From raw PTA residuals → parameter estimation → posterior inference

---

## 🏗️ Architecture

### Stage 1: Physics-Informed Transformer

- Input: PTA time series
- Patch-based tokenization or CNN stem
- Positional + physics-informed encodings
- External-attention Transformer backbone
- Outputs:
  - Point estimates
  - Latent representation

### Stage 2: Posterior Inference

- Conditioned on Transformer features
- Learns full posterior:

$$p(\theta \mid x)$$

- Supports:
  - DNF (fast, exact likelihood)
  - CNF (continuous, expressive)

---

## 📊 Dataset

- Synthetic PTA dataset:
  - ~10 pulsars
  - Multiple noise realizations
- Target parameters:
  - Orbital frequency
  - Eccentricity
  - Chirp mass
  - Signal amplitude
- Controlled SNR range (e.g. 20–30)

### Modes

- **Flattened mode** → single pulsar per sample
- **Realisation mode** → full PTA dataset

---

## 📂 Repository Structure

```
PIPE_GWs_PTA/
│
├── flattened/
│   ├── notebooks/          # Experiments and training
│   ├── package/            # Core models
│   └── best_fast.pt        # Pretrained checkpoint
│
├── realisation/            # Multi-pulsar dataset mode
│
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/subhajitphy/PIPE_GWs_PTA.git
cd PIPE_GWs_PTA
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Transformer (point estimation)

```bash
python pt_est.py
```

### Train Normalizing Flows

```bash
python run_dnfs.py
# or
python run_cnfs.py
```

### Evaluation

```bash
python train_plot_eval.py
```

---

## 📈 Results

- Improved parameter recovery vs baseline Transformers
- Sharper posterior distributions
- Robust performance in low-SNR regimes
- Significant speed-up over traditional Bayesian methods

---

## 📖 Citation

```bibtex
@inproceedings{dandapat2026pipe,
  title={Transformers with Physics-informed encodings and Simulation-Based inference for robust Gravitational-Wave detection in Pulsar Timing Array data},
  author={Dandapat, Subhajit and Chua, Alvin J. K.},
  booktitle={AI4X – Accelerate Conference},
  year={2026},
  url={https://openreview.net/forum?id=Bpo4HgB6by}
}
```

---

## 🤝 Acknowledgements

- National University of Singapore (NUS)
- PTA collaborations (IPTA, NANOGrav, EPTA, InPTA)

---

## 📬 Contact

**Subhajit Dandapat**
Postdoctoral Research Fellow, NUS
📧 subhajit.phy97@gmail.com
