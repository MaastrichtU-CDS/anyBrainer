"""FOMO25 inference script for anyBrainer models."""

import argparse
from pathlib import Path
import os

from typing import Literal

from .preprocess import preprocess_inputs
from anyBrainer.core.data import (
    ClassificationDataModule,
)
from anyBrainer.core.engines import (
    ClassificationModel,
)

Task = Literal["task1", "task2", "task3"]

task_1_config = {
    "pl_module_settings": {
        "name": "ClassificationModel",
        "model_kwargs": {
            "name": "Swinv2Classifier",
            "model_kwargs": {"num_classes": 1},
        },
    },
}

def predict_task_1():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flair", required=True)
    ap.add_argument("--adc", required=True)
    ap.add_argument("--dwi_b1000", required=True)
    ap.add_argument("--swi", default=None)
    ap.add_argument("--t2s", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    work_dir = Path(os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))

    preprocess_inputs(
        inputs=[args.flair, args.adc, args.dwi_b1000, args.t2s, args.swi],
        mods=["flair", "adc", "dwi_b1000", "swi", "t2s"],
        task="task1",
        work_dir=work_dir,
    )
    
    # Get model
    model = ClassificationModel(**task_1_config["pl_module_settings"])

    # TODO:
    # 1. Get predict transforms and apply to input paths. 
    # 2. Load model weights; do when instantiating pl_module
    # 3. Predict; single pass.
    # 4. Save predictions to output_dir. 





    
