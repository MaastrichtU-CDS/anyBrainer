# anyBrainer 🧠
**anyBrainer** is a PyTorch Lightning + MONAI–based framework for **pretraining and finetuning foundation models** for brain MRI analysis.  
It extends MONAI and Lightning with custom transforms, optimizers/schedulers, and inferers to provide a flexible end-to-end pipeline for research and challenge participation.

The framework supports **contrastive learning pretraining** as well as downstream tasks:
- **Classification** (e.g., brain infarct detection)
- **Segmentation** (e.g., meningioma segmentation)
- **Regression** (e.g., brain age prediction)
- **Multimodal fusion** setups

anyBrainer is actively used for the [MICCAI FOMO25 challenge](https://fomo25.grand-challenge.org/) across **all tasks**:
- Pretraining  
- Finetuning for classification, segmentation, and regression  
- Inference and containerized submission

---

## Features ✨
- 🔧 **Config-driven workflows**: Run pretraining/finetuning/inference via `.yaml` configs  
  (`pretrain_cl.yaml`, `finetune_cls.yaml`, `finetune_seg.yaml`, `finetune_reg.yaml`)  
- 🌀 **Custom MONAI transforms** for MRI-specific augmentation and preprocessing  
- 🧩 **Multi-param optimizer & scheduler factories** for flexible training setups  
- 📦 **Custom inferers** for efficient sliding-window and multimodal inference  
- 🔄 **Contrastive learning setup** for foundation model pretraining  
- ⚡ Built on **PyTorch Lightning** for reproducibility and scalability  
- 🏗️ **Containerization-ready**: inference scripts and [Apptainer](https://apptainer.org/) specs in [`app/`](./app)  

---

## Installation 🚀

Clone the repository and install via `pyproject.toml`:

```bash
# Install basic dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For testing
pip install -e ".[test]"

# For all dependencies
pip install -e ".[dev,test]"
