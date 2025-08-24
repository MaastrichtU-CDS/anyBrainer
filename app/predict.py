"""FOMO25 inference script for anyBrainer models."""

import argparse
from pathlib import Path
import os
from typing import Literal, cast
import logging
from copy import deepcopy

import torch
from monai.transforms.compose import Compose
from monai.transforms.post.dictionary import Invertd
from monai.data.utils import list_data_collate, decollate_batch

from .preprocess import preprocess_inputs, revert_preprocess
from .utils import write_probability, download_templates

from anyBrainer.core.engines import (
    ClassificationModel,
)
from anyBrainer.factories import UnitFactory

Task = Literal["task1", "task2", "task3"]

TASK_1_CONFIG = {
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
        "ckpts/task1/split_0_last.ckpt",
        "ckpts/task1/split_1_last.ckpt",
        "ckpts/task1/split_2_last.ckpt",
        "ckpts/task1/split_3_last.ckpt",
        "ckpts/task1/split_4_last.ckpt",
    ]
}

TASK_2_CONFIG = {
    "predict_transforms": {
        "name": "get_predict_transforms",
        "keys": ["dwi", "flair", "swi", "t2s"],
        "allow_missing_keys": True,
        "is_nifti": True,
        "concat_img": True,
        "delete_orig": False,
        "sliding_window": False,
        "target_key": "img"
    },
    "postprocess_transforms": {
        "name": "get_postprocess_segmentation_transforms",
    },
    "model_ckpts": [
        "ckpts/task2/split_0_last.ckpt",
        "ckpts/task2/split_1_last.ckpt",
        "ckpts/task2/split_2_last.ckpt",
        "ckpts/task2/split_3_last.ckpt",
        "ckpts/task2/split_4_last.ckpt",
    ]
}

TASK_3_CONFIG = {
    "predict_transforms": {
        "name": "get_predict_transforms",
        "keys": ["t1", "t2"],
        "allow_missing_keys": False,
        "is_nifti": True,
        "concat_img": True,
        "sliding_window": True,
        "target_key": "img"
    },
    "model_ckpts": [
        "ckpts/task3/split_0_last.ckpt",
        "ckpts/task3/split_1_last.ckpt",
        "ckpts/task3/split_2_last.ckpt",
        "ckpts/task3/split_3_last.ckpt",
        "ckpts/task3/split_4_last.ckpt",
    ]
}

def predict_task_1():
    """
    Inference pipeline for task 1 - infarct classification.
    """
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
    elif args.swi is not None and args.t2s is None:
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / args.swi.name
    elif args.swi is None and args.t2s is not None:
        opt_mod_key = "t2s"
        opt_mod_path = work_dir / "inputs" / args.t2s.name
    else:
        raise ValueError("Either SWI or T2* modality must be provided.")

    preprocess_inputs(
        inputs=[args.flair, args.adc, args.dwi_b1000, args.swi, args.t2s],
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

    # Load transforms
    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_1_CONFIG["predict_transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_1_CONFIG["postprocess_transforms"]))

    # Load ensemble
    models = []
    for ck in TASK_1_CONFIG["model_ckpts"]:
        m = ClassificationModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 1.")

    # Predict (mean logits)
    with torch.no_grad():
        batch = list_data_collate([predict_transforms(input_dict)])  # dict->batched dict
        mean_logits: torch.Tensor | None = None
        w = 1.0 / max(1, len(models))
        for m in models:
            curr_input = cast(dict[str, torch.Tensor], deepcopy(batch))
            logits = cast(
                torch.Tensor,
                m.predict(curr_input, img_key="img", do_postprocess=False, invert=False),
            )
            mean_logits = logits * w if mean_logits is None else mean_logits + logits * w

        # Optional postprocess; assume it returns a scalar probability tensor
        out = cast(torch.Tensor, postprocess_transforms(mean_logits))
        prob = float(out.item())

    # Save
    write_probability(args.output, prob)

def predict_task_2():
    """
    Inference pipeline for task 2 - meningioma segmentation.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--flair", required=True)
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
    elif args.swi is not None and args.t2s is None:
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / args.swi.name
    elif args.swi is None and args.t2s is not None:
        opt_mod_key = "t2s"
        opt_mod_path = work_dir / "inputs" / args.t2s.name
    else:
        raise ValueError("Either SWI or T2* modality must be provided.")

    preprocess_inputs(
        inputs=[args.dwi_b1000, args.flair, args.swi, args.t2s],
        mods=["dwi_b1000", "flair", "swi", "t2s"],
        task="task2",
        work_dir=work_dir,
    )

    input_dict = {
        "dwi": work_dir / "inputs" / args.dwi_b1000.name,
        "flair": work_dir / "inputs" / args.flair.name,
        opt_mod_key: opt_mod_path,
    }

    for p in input_dict.values():
        if not p.exists():
            raise FileNotFoundError(f"Cannot find preprocessed input {p.name}; "
                                    f"check working directory {work_dir} or any "
                                    f"preprocessing failures.")
    # Load transforms
    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_2_CONFIG["predict_transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_2_CONFIG["postprocess_transforms"]))

    # Load ensemble
    models = []
    for ck in TASK_2_CONFIG["model_ckpts"]:
        m = ClassificationModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 2.")

    # Predict (mean logits)
    with torch.no_grad():
        batch = cast(dict[str, torch.Tensor], list_data_collate([predict_transforms([input_dict])]))
        mean_logits: torch.Tensor | None = None
        w = 1.0 / max(1, len(models))
        for m in models:
            curr_input = cast(dict[str, torch.Tensor], deepcopy(batch))
            logits = cast(
                torch.Tensor,
                m.predict(curr_input, img_key="img", do_postprocess=False, invert=True),
            )
            mean_logits = logits * w if mean_logits is None else mean_logits + logits * w

        # Optional postprocess; assume it returns a scalar probability tensor
        batch['pred'] = cast(torch.Tensor, postprocess_transforms(mean_logits))

    # Revert prediction
    inv = Invertd(keys=["pred"], orig_keys=["flair"], transform=predict_transforms)
    inv_batch = cast(
        dict[str, torch.Tensor],
        list_data_collate([inv(d) for d in cast(list[dict], decollate_batch(batch))])
    )
    pred_img = revert_preprocess(inv_batch['pred'], input_dict['flair'], work_dir)

    # Save
    pred_img.to_file(args.output)

def predict_task_3():
    """
    Inference pipeline for task 3 - brain age prediction.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    work_dir = Path(os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))

    preprocess_inputs(
        inputs=[args.t1, args.t2],
        mods=["t1", "t2"],
        task="task3",
        work_dir=work_dir,
    )

    input_dict = {
        "t1": work_dir / "inputs" / args.t1.name,
        "t2": work_dir / "inputs" / args.t2.name,
    }

    for p in input_dict.values():
        if not p.exists():
            raise FileNotFoundError(f"Cannot find preprocessed input {p.name}; "
                                    f"check working directory {work_dir} or any "
                                    f"preprocessing failures.")
    # Load transforms
    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_3_CONFIG["predict_transforms"]))

    # Load ensemble
    models = []
    for ck in TASK_3_CONFIG["model_ckpts"]:
        m = ClassificationModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 3.")

    # Predict (mean logits)
    with torch.no_grad():
        batch = list_data_collate([predict_transforms(input_dict)])  # dict->batched dict
        mean_logits: torch.Tensor | None = None
        w = 1.0 / max(1, len(models))
        for m in models:
            curr_input = cast(dict[str, torch.Tensor], deepcopy(batch))
            logits = cast(
                torch.Tensor,
                m.predict(curr_input, img_key="img", do_postprocess=False, invert=False),
            )
            mean_logits = logits * w if mean_logits is None else mean_logits + logits * w

        # Optional postprocess; assume it returns a scalar probability tensor
        out = float(cast(torch.Tensor, mean_logits).item())

    # Save
    write_probability(args.output, out)

if __name__ == "__main__":
    download_templates()
    predict_task_1()