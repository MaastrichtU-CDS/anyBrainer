# anyBrainer 🧠
**anyBrainer** is a PyTorch Lightning + MONAI–based framework for **pretraining and finetuning foundation models** for brain MRI analysis.  

It extends MONAI and Lightning with custom transforms, optimizers/schedulers, and inferers to provide a flexible end-to-end pipeline for a wide range of downstream tasks.

The framework supports **contrastive learning pretraining** as well as downstream tasks:
- **Classification** (e.g., brain infarct detection)
- **Segmentation** (e.g., meningioma segmentation)
- **Regression** (e.g., brain age prediction)
- **Multimodal fusion** setups

**anyBrainer** was used for the [MICCAI FOMO25 challenge](https://fomo25.github.io./) across **all tasks**:
- Pretraining  
- Finetuning for classification, segmentation, and regression  
- Inference and containerized submission


## Features
- **Config-driven workflows**: Run pretraining/finetuning/inference via `.yaml` configs  
  (`pretrain_cl.yaml`, `finetune_cls.yaml`, `finetune_seg.yaml`, `finetune_reg.yaml`)  
- **Custom MONAI transforms** for MRI-specific augmentation and preprocessing  
- **Multi-param optimizer & scheduler factories** for flexible training setups  
- **Custom inferers** for efficient sliding-window and multimodal inference  
- **Contrastive learning setup** for foundation model pretraining  
- Built on **PyTorch Lightning** for reproducibility and scalability
- Compatible with **Weights & Biases** logging for detailed experiment tracking
- **Containerization-ready**: inference scripts and [Apptainer](https://apptainer.org/) specs in [`app/`](./app)  

## NEW features (2026)
- **Masked Image modeling setup** for pretraining
- **Multimodal** datamodule and model for flexible modality-aware data loading and training
- **Multi-level sweep workflow** for performing multiple finetuning (and more) experiments at once


## Installation

Clone the repository and install via:

```bash
# For basic dependencies:
pip install -e .

# For all dependencies:
pip install -e ".[dev,test]"
```


## Usage

Training and inference are fully config-driven. Example configs are provided in the repo.

```bash
# Pretraining MIM (NEW)
anyBrainer TrainWorkflow "pretrain_mim.yaml"

# Pretraining contrastive learning
anyBrainer TrainWorkflow "pretrain_cl.yaml"

# Finetuning for classification
anyBrainer CVWorkflow "finetune_cls.yaml"

# Finetuning for segmentation
anyBrainer CVWorkflow "finetune_seg.yaml"

# Finetuning for regression
anyBrainer CVWorkflow "finetune_reg.yaml"

# Finetuning for multimodal tasks (NEW)
anyBrainer CVWorkflow "finetune_mm.yaml"

# Finetune and evaluate multiple models at once (NEW)
anyBrainer SweepWorkflow "finetune_sweep.yaml"
```


## Project Structure

```text
anyBrainer/
├── app/                  # Inference scripts and Apptainer files (containerization)
├── src/                  # Source code
│   └── anyBrainer/       # Main package
│       ├── config/       # Config management
│       ├── core/         # Core modules: data, engines, networks, losses, transforms, etc.
│       ├── factories/    # Factories for models, optimizers, schedulers
│       ├── interfaces/   # Base classes and interfaces
│       ├── log/          # Logging utilities
│       ├── registry/     # Registries for models, transforms, losses, etc.
│       ├── __main__.py   # CLI entry point
│       └── __init__.py
├── tests/                # Unit and integration tests
├── pretrain_*.yaml       # Example configs for pretraining
├── finetune_*.yaml       # Example configs for finetuning
├── pyproject.toml        # Project dependencies
├── LICENSE
└── README.md
```

## Citation

If you use anyBrainer in your research, please cite this repository:

```bibtex
@misc{anybrainer2025,
  title        = {anyBrainer: A PyTorch Lightning + MONAI Framework for Brain MRI Pretraining and Finetuning},
  author       = {Petros Koutsouvelis},
  year         = {2025},
  url          = {https://github.com/MaastrichtU-CDS/anyBrainer},
  note         = {GitHub repository}
}
```



## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
