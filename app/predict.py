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

from preprocess import preprocess_inputs, revert_preprocess
from utils import (
    write_probability, 
    move_batch_to_device,
    get_device,
)

from anyBrainer.core.engines import (
    ClassificationModel,
    RegressionModel,
    SegmentationModel,
)
from anyBrainer.factories import UnitFactory

Task = Literal["task1", "task2", "task3"]

TEMPL_DIR = Path(os.getenv("ANYBRAINER_TEMPLATES_DIR", "/opt/anyBrainer/templates"))
CKPTS_DIR = Path(os.getenv("ANYBRAINER_CKPTS_DIR", "/opt/anyBrainer/ckpts"))

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
        "task1/run_0.ckpt",
        "task1/run_1.ckpt",
        "task1/run_2.ckpt",
        "task1/run_3.ckpt",
        "task1/run_4.ckpt",
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
        "task2/run_0.ckpt",
        "task2/run_1.ckpt",
        "task2/run_2.ckpt",
        "task2/run_3.ckpt",
        "task2/run_4.ckpt",
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
        "task3/run_0.ckpt",
        "task3/run_1.ckpt",
        "task3/run_2.ckpt",
        "task3/run_3.ckpt",
        "task3/run_4.ckpt",
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
        opt_mod_path = work_dir / "inputs" / Path(args.swi).name
    elif args.swi is not None and args.t2s is None:
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / Path(args.swi).name
    elif args.swi is None and args.t2s is not None:
        opt_mod_key = "t2s"
        opt_mod_path = work_dir / "inputs" / Path(args.t2s).name
    else:
        raise ValueError("Either SWI or T2* modality must be provided.")

    preprocess_inputs(
        inputs=[args.flair, args.adc, args.dwi_b1000, args.swi, args.t2s],
        mods=["flair", "adc", "dwi_b1000", "swi", "t2s"],
        ref_mod="flair",
        work_dir=work_dir,
        tmpl_path=TEMPL_DIR / "icbm_mni152_t1_09a_asym_bet.nii.gz",
    )

    input_dict = {
        "flair": work_dir / "inputs" / Path(args.flair).name,
        "dwi": work_dir / "inputs" / Path(args.dwi_b1000).name,
        "adc": work_dir / "inputs" / Path(args.adc).name,
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
    ckpts = [CKPTS_DIR / p for p in TASK_1_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")

        m = ClassificationModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 1.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = list_data_collate([predict_transforms(input_dict)])
    with torch.no_grad():
        for m in models:
            m = m.to(device).eval()
            try:
                m.freeze()
            except Exception:
                pass
            curr = move_batch_to_device(cast(dict[str, torch.Tensor], deepcopy(cpu_batch)), device)
            logits = m.predict(curr, img_key="img", do_postprocess=False, invert=False)
            logits_cpu = logits.detach().float().cpu()

            mean_logits = logits_cpu * w if mean_logits is None else mean_logits + logits_cpu * w

            m = m.to("cpu")
            torch.cuda.empty_cache()

        # Postprocess
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
        opt_mod_path = work_dir / "inputs" / Path(args.swi).name
    elif args.swi is not None and args.t2s is None:
        opt_mod_key = "swi"
        opt_mod_path = work_dir / "inputs" / Path(args.swi).name
    elif args.swi is None and args.t2s is not None:
        opt_mod_key = "t2s"
        opt_mod_path = work_dir / "inputs" / Path(args.t2s).name
    else:
        raise ValueError("Either SWI or T2* modality must be provided.")

    preprocess_inputs(
        inputs=[args.dwi_b1000, args.flair, args.swi, args.t2s],
        mods=["dwi_b1000", "flair", "swi", "t2s"],
        ref_mod="flair",
        work_dir=work_dir,
        tmpl_path=TEMPL_DIR / "icbm_mni152_t2_09a_asym_bet.nii.gz",
    )

    input_dict = {
        "dwi": work_dir / "inputs" / Path(args.dwi_b1000).name,
        "flair": work_dir / "inputs" / Path(args.flair).name,
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
    ckpts = [CKPTS_DIR / p for p in TASK_2_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")

        m = SegmentationModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 2.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = cast(dict[str, torch.Tensor], list_data_collate([predict_transforms(input_dict)]))
    with torch.no_grad():
        for m in models:
            m = m.to(device).eval()
            try:
                m.freeze()
            except Exception:
                pass
            curr = move_batch_to_device(cast(dict[str, torch.Tensor], deepcopy(cpu_batch)), device)
            logits = m.predict(curr, img_key="img", do_postprocess=False, invert=True)
            logits_cpu = logits.detach().float().cpu()
            mean_logits = logits_cpu * w if mean_logits is None else mean_logits + logits_cpu * w

            m = m.to("cpu")
            torch.cuda.empty_cache()
    
    # Postprocess
    cpu_batch['pred'] = cast(torch.Tensor, postprocess_transforms(mean_logits))

    # Revert prediction
    inv = Invertd(
        keys="pred", 
        orig_keys="flair", 
        transform=predict_transforms, 
        nearest_interp=True,
    )
    inv_batch = cast(
        dict[str, torch.Tensor],
        list_data_collate([inv(d) for d in cast(list[dict], decollate_batch(cpu_batch))])
    )
    pred_img = revert_preprocess(inv_batch['pred'][0], args.flair, work_dir)

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
        ref_mod="t1",
        work_dir=work_dir,
        tmpl_path=TEMPL_DIR / "icbm_mni152_t1_09a_asym_bet.nii.gz",
    )

    input_dict = {
        "t1": work_dir / "inputs" / Path(args.t1).name,
        "t2": work_dir / "inputs" / Path(args.t2).name,
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
    ckpts = [CKPTS_DIR / p for p in TASK_3_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")

        m = RegressionModel.load_from_checkpoint(ck, map_location="cpu")
        m.eval()
        try:
            m.freeze()  # if LightningModule
        except Exception:
            pass
        models.append(m)
    logging.info(f"Loaded {len(models)} models for task 3.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = list_data_collate([predict_transforms(input_dict)])
    with torch.no_grad():
        for m in models:
            m = m.to(device).eval()
            try:
                m.freeze()
            except Exception:
                pass
            curr = move_batch_to_device(cast(dict[str, torch.Tensor], deepcopy(cpu_batch)), device)
            logits = m.predict(curr, img_key="img", do_postprocess=False, invert=False)
            logits_cpu = logits.detach().float().cpu()

            mean_logits = logits_cpu * w if mean_logits is None else mean_logits + logits_cpu * w

            m = m.to("cpu")
            torch.cuda.empty_cache()

        # Optional postprocess; assume it returns a scalar probability tensor
        out = float(cast(torch.Tensor, mean_logits).item())

    # Save
    write_probability(args.output, out)

def main():
    """
    Main function to run the inference pipeline.
    """
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    predict_task_1()

if __name__ == "__main__":
    main()