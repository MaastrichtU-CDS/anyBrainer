"""FOMO25 inference script for anyBrainer models."""

import argparse
from pathlib import Path
import os
from typing import Literal, cast
import logging
from copy import deepcopy

import torch
from monai.transforms.compose import Compose
from monai.data.utils import list_data_collate

from .preprocess import preprocess_inputs
from .utils import write_probability

from anyBrainer.core.engines import (
    ClassificationModel,
)
from anyBrainer.factories import UnitFactory

Task = Literal["task1", "task2", "task3"]

task_1_config = {
    "predict_transforms": {
        "name": "get_predict_transforms",
        "keys": ["flair", "dwi", "adc", "swi", "t2s"],
        "allow_missing_keys": True,
        "is_nifti": True,
        "concat_img": True,
        "sliding_window": True,
        "target_key": "img"
    },
    "postprocess_transforms": {
        "name": "get_postprocess_classification_transforms",
    },
    "model_ckpts": [
        "ckpts/task1/split_0/last.ckpt",
        "ckpts/task1/split_1/last.ckpt",
        "ckpts/task1/split_2/last.ckpt",
        "ckpts/task1/split_3/last.ckpt",
        "ckpts/task1/split_4/last.ckpt",
    ]
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

    if args.swi is not None and args.t2s is not None:
        logging.warning("Both SWI and T2* modalities provided; using SWI only.")
        args.t2s = None
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / args.swi.name
    elif args.swi is not None and args.swi is None:
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / args.swi.name
    elif args.swi is None and args.t2s is not None:
        opt_mod_key = "t2s"
        opt_mod_path = work_dir / "inputs" / args.t2s.name
    else:
        raise ValueError("Either SWI or T2* modality must be provided.")

    preprocess_inputs(
        inputs=[args.flair, args.adc, args.dwi_b1000, args.t2s, args.swi],
        mods=["flair", "adc", "dwi_b1000", "swi", "t2s"],
        task="task1",
        work_dir=work_dir,
    )

    input_dict = {
        "flair": work_dir / "inputs" / args.flair.name,
        "dwi": work_dir / "inputs" / args.dwi_b1000.name,
        "adc": work_dir / "inputs" / args.adc.name,
        opt_mod_key: opt_mod_path,
    }

    for p in input_dict.values():
        if not p.exists():
            raise FileNotFoundError(f"Cannot find preprocessed input {p.name}; "
                                    f"check working directory {work_dir} or any "
                                    f"preprocessing failures.")

    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(task_1_config["transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(task_1_config["postprocess_transforms"]))
    models = [ClassificationModel.load_from_checkpoint(ckpt) for ckpt in task_1_config["model_ckpts"]]

    # Predict
    model_input = list_data_collate([predict_transforms(input_dict)])
    mean_logits: torch.Tensor | None = None
    w = 1 / len(models)
    for model in models:
        curr_input = cast(dict[str, torch.Tensor], deepcopy(model_input))
        if mean_logits is None:
            mean_logits = w *cast(
                torch.Tensor, model.predict(curr_input, img_key="img", do_postprocess=False, invert=False)
            )
        else:
            mean_logits = mean_logits + w * cast(
                torch.Tensor, model.predict(curr_input, img_key="img", do_postprocess=False, invert=False)
            )
    out = cast(torch.Tensor, postprocess_transforms(mean_logits))

    # Save
    write_probability(args.output, out.item())







    
