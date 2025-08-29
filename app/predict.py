"""FOMO25 inference script for anyBrainer models."""

import argparse
from pathlib import Path
import os
from typing import Literal, cast
import logging
from copy import deepcopy
from tqdm import tqdm

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
from anyBrainer.factories import UnitFactory, ModuleFactory

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
    "pl_module_settings": {
        "name": "ClassificationModel",
        "model_kwargs": {
            "name": "Swinv2Classifier",
            "patch_size": 2,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "feature_size": 48,
            "use_v2": True,
            "extra_swin_kwargs": {
                "use_checkpoint": True,
            },
            "mlp_num_classes": 1,
            "mlp_num_hidden_layers": 1,
            "mlp_hidden_dim": 128,
            "late_fusion": True,
            "n_late_fusion": 4
        },
        "optimizer_kwargs": {
            "name": "AdamW",
            "auto_no_weight_decay": True,
            "param_groups": [
                {
                    "lr": 0.0005,
                    "weight_decay": 0.001,
                    "param_group_prefix": ["classification_head", "fusion_head"]
                },
                {
                    "lr": 0.00002,
                    "weight_decay": 0.00005,
                    "param_group_prefix": ["encoder.layers4"]
                }
            ]
        },
        "lr_scheduler_kwargs": {
            "name": "CosineAnnealingWithWarmup",
            "interval": "step",
            "frequency": 1,
            "warmup_iters": [30, 18], # [10%, 20%]
            "start_iter": [0, 210],
            "eta_min": [0.00005, 0.000002],
            "total_iters": 300
        },
        "loss_fn_kwargs": {
            "name": "BCEWithLogitsLoss",
        },
        "weights_init_settings": {
            "weights_init_fn": None,
            "load_pretrain_weights": None, # populated in the main function
            "load_param_group_prefix": None,
            "rename_map": {
                "model.": ""
            }
        },
        "inference_settings": {
            "postprocess": None,
            "tta": "get_flip_tta"
        }
    },
    "postprocess_transforms": {
        "name": "get_postprocess_classification_transforms",
        "as_discrete": False,
    },
    "model_ckpts": [
        "run_0.ckpt",
        "run_1.ckpt",
        "run_2.ckpt",
        "run_3.ckpt",
        "run_4.ckpt",
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
        "target_key": "img",
        "ref_key": "flair"
    },
    "pl_module_settings": {
        "name": "SegmentationModel",
        "model_kwargs": {
            "name": "Swinv2LateFusionFPNDecoder",
            "in_channels": 1,
            "out_channels": 1,
            "patch_size": 2,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "feature_size": 48,
            "use_v2": True,
            "extra_swin_kwargs": {
                "use_checkpoint": True,
            },
            "n_late_fusion": 3
        },
        "optimizer_kwargs": {
            "name": "AdamW",
            "auto_no_weight_decay": True,
            "param_groups": [
                {
                    "lr": 0.0005,
                    "weight_decay": 0.0001,
                    "param_group_prefix": "decoder"
                },
                {
                    "lr": 0.00002,
                    "weight_decay": 0.00001,
                    "param_group_prefix": ["encoder.layers4"]
                }
            ]
        },
        "lr_scheduler_kwargs": {
            "name": "CosineAnnealingWithWarmup",
            "interval": "step",
            "frequency": 1,
            "warmup_iters": [144, 81], # [8%, 15%]
            "start_iter": [0, 1260],
            "eta_min": [0.00005, 0.000002],
            "total_iters": 1800
        },
        "loss_fn_kwargs": {
            "name": "monai:DiceCELoss",
            "sigmoid": True,
        },
        "weights_init_settings": {
            "weights_init_fn": None,
            "load_pretrain_weights": None, # populated in the main function
            "load_param_group_prefix": None,
            "rename_map": {
                "model.": ""
            }
        },
        "inference_settings": {
            "inferer_kwargs": {
                "name": "SlidingWindowInferer",
                "roi_size": 128,
                "mode": "constant",
                "overlap": 0.5,
            },
            "postprocess": None,
            "tta": "get_flip_tta"
        }
    },
    "postprocess_transforms": {
        "name": "get_postprocess_segmentation_transforms",
        "keep_largest": False,
        "remove_small_objects": False,
    },
    "model_ckpts": [
        "run_0.ckpt",
        "run_1.ckpt",
        "run_2.ckpt",
        "run_3.ckpt",
        "run_4.ckpt",
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
    "pl_module_settings": {
        "name": "RegressionModel",
        "model_kwargs": {
            "name": "Swinv2Classifier",
            "patch_size": 2,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "feature_size": 48,
            "use_v2": True,
            "extra_swin_kwargs": {
                "use_checkpoint": True,
            },
            "mlp_num_classes": 1,
            "mlp_num_hidden_layers": 1,
            "mlp_hidden_dim": 128,
            "mlp_dropout": 0.2,
            "late_fusion": True,
            "n_late_fusion": 2
        },
        "optimizer_kwargs": {
            "name": "AdamW",
            "auto_no_weight_decay": True,
            "param_groups": [
                {
                    "lr": 0.0005,
                    "weight_decay": 0.001,
                    "param_group_prefix": ["fusion_head", "classification_head"]
                },
                {
                    "lr": 0.0001,
                    "weight_decay": 0.00005,
                    "param_group_prefix": "encoder.layers4"
                },
                {
                    "lr": 0.00007,
                    "weight_decay": 0.00005,
                    "param_group_prefix": "encoder.layers3"
                },
                {
                    "lr": 0.00005,
                    "weight_decay": 0.00005,
                    "param_group_prefix": "encoder.layers2"
                },
                {
                    "lr": 0.00003,
                    "weight_decay": 0.00005,
                    "param_group_prefix": "encoder.layers1"
                }
            ]
        },
        "lr_scheduler_kwargs": {
            "name": "CosineAnnealingWithWarmup",
            "interval": "step",
            "frequency": 1,
            "warmup_iters": [0.00005, 0.00001, 0.000007, 0.000005, 0.000003],
            "start_iter": [96, 116, 126, 108, 80], # [8%, 12%, 15%, 18%, 22%]
            "eta_min": [0, 240, 360, 600, 840],
            "total_iters": 1200,
        },
        "loss_fn_kwargs": {
            "name": "SmoothL1Loss",
            "beta": 0.1,
        },
        "weights_init_settings": {
            "weights_init_fn": None,
            "load_pretrain_weights": None, # populated in the main function
            "load_param_group_prefix": None,
            "rename_map": {
                "model.": ""
            }
        },
        "center_labels": "fixed:65",
        "scale_labels": [20, 90],
        "bias_init": None,
    },
    "postprocess_transforms": {
        "name": "get_postprocess_regression_transforms",
        "center": 65,
        "scale_range": [20, 90],
    },
    "model_ckpts": [
        "run_0.ckpt",
        "run_1.ckpt",
        "run_2.ckpt",
        "run_3.ckpt",
        "run_4.ckpt",
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
    ap.add_argument("--debug", default=False, action="store_true")
    args = ap.parse_args()

    logging.info(f"====RUNNING BRAIN INFARCT CLASSIFICATION PIPELINE====\n")
    if not args.debug:
        logging.getLogger("anyBrainer").setLevel(logging.WARNING)

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

    logging.info(f"Preprocessing inputs...")
    preprocess_inputs(
        inputs=[args.flair, args.adc, args.dwi_b1000, args.swi, args.t2s],
        mods=["flair", "adc", "dwi_b1000", "swi", "t2s"],
        ref_mod="flair",
        work_dir=work_dir,
        tmpl_path=TEMPL_DIR / "icbm_mni152_t2_09a_asym_bet.nii.gz",
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
    logging.info(f"Successfully preprocessed inputs.")

    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_1_CONFIG["predict_transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_1_CONFIG["postprocess_transforms"]))

    # Load ensemble
    models = []
    ckpts = [CKPTS_DIR / p for p in TASK_1_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")
        model_cfg = deepcopy(TASK_1_CONFIG["pl_module_settings"])
        model_cfg["weights_init_settings"]["load_pretrain_weights"] = ck
        models.append(ModuleFactory.get_pl_module_instance_from_kwargs(model_cfg))
    logging.info(f"Loaded {len(models)} models for task 1.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = list_data_collate([predict_transforms(input_dict)])
    with torch.no_grad():
        logging.info(f"Running ensemble of {len(models)} models...")
        for m in tqdm(models):
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

    logging.info(f"Ensemble prediction completed with probability {prob}. "
                 f"Writing output to {args.output}...")

    # Save
    write_probability(args.output, prob)
    logging.info(f"Output written to {args.output}.")
    logging.info(f"====BRAIN INFARCT CLASSIFICATION PIPELINE COMPLETED====\n")

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
    ap.add_argument("--debug", default=False, action="store_true")
    args = ap.parse_args()

    logging.info(f"====RUNNING BRAIN MENINGIOMA SEGMENTATION PIPELINE====\n")
    if not args.debug:
        logging.getLogger("anyBrainer").setLevel(logging.WARNING)

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

    logging.info(f"Preprocessing inputs...")
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
    logging.info(f"Successfully preprocessed inputs.")

    # Load transforms
    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_2_CONFIG["predict_transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_2_CONFIG["postprocess_transforms"]))

    # Load ensemble
    models = []
    ckpts = [CKPTS_DIR / p for p in TASK_2_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")
        model_cfg = deepcopy(TASK_2_CONFIG["pl_module_settings"])
        model_cfg["weights_init_settings"]["load_pretrain_weights"] = ck
        models.append(ModuleFactory.get_pl_module_instance_from_kwargs(model_cfg))
    logging.info(f"Loaded {len(models)} models for task 2.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = cast(dict[str, torch.Tensor], list_data_collate([predict_transforms(input_dict)]))
    with torch.no_grad():
        logging.info(f"Running ensemble of {len(models)} models...")
        for m in tqdm(models):
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
    logging.info(f"Reverting predicted segmentation mask to original space...")
    inv = Invertd(
        keys="pred", 
        orig_keys="flair", 
        transform=predict_transforms, 
        nearest_interp=True,
    )
    logging.info(f"Reverting cropping, resampling, and orientation transforms...")
    inv_batch = cast(
        dict[str, torch.Tensor],
        list_data_collate([inv(d) for d in cast(list[dict], decollate_batch(cpu_batch))])
    )
    logging.info(f"Cropping, resampling, and orientation transforms reverted; "
                 f"reverting registration to template...")
    pred_img = revert_preprocess(inv_batch['pred'][0], args.flair, work_dir, 
                                 tmpl_path=TEMPL_DIR / "icbm_mni152_t2_09a_asym_bet.nii.gz")
    logging.info(f"Image reverted to original space; saving to {args.output}...")

    # Save
    pred_img.to_file(args.output)
    logging.info(f"Output written to {args.output}.")
    logging.info(f"====BRAIN MENINGIOMA SEGMENTATION PIPELINE COMPLETED====\n")

def predict_task_3():
    """
    Inference pipeline for task 3 - brain age prediction.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--debug", default=False, action="store_true")
    args = ap.parse_args()

    logging.info(f"====RUNNING BRAIN AGE PREDICTION PIPELINE====\n")
    if not args.debug:
        logging.getLogger("anyBrainer").setLevel(logging.WARNING)

    work_dir = Path(os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))

    logging.info(f"Preprocessing inputs...")
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
    logging.info(f"Successfully preprocessed inputs.")

    # Load transforms
    predict_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_3_CONFIG["predict_transforms"]))
    postprocess_transforms = Compose(UnitFactory.get_transformslist_from_kwargs(TASK_3_CONFIG["postprocess_transforms"]))

    # Load ensemble
    models = []
    ckpts = [CKPTS_DIR / p for p in TASK_3_CONFIG["model_ckpts"]]
    for ck in ckpts:
        if not ck.exists():
            raise FileNotFoundError(f"Cannot find requested checkpoint {ck.name}.")
        model_cfg = deepcopy(TASK_3_CONFIG["pl_module_settings"])
        model_cfg["weights_init_settings"]["load_pretrain_weights"] = ck
        models.append(ModuleFactory.get_pl_module_instance_from_kwargs(model_cfg))
    logging.info(f"Loaded {len(models)} models for task 3.")

    # Predict (mean logits)
    device = get_device()
    mean_logits = None
    w = 1.0 / max(1, len(models))
    cpu_batch = list_data_collate([predict_transforms(input_dict)])
    with torch.no_grad():
        logging.info(f"Running ensemble of {len(models)} models...")
        for m in tqdm(models):
            m = m.to(device).eval()
            try:
                m.freeze()
            except Exception:
                pass
            curr = move_batch_to_device(cast(dict[str, torch.Tensor], deepcopy(cpu_batch)), device)
            logits = m.predict(curr, img_key="img", do_postprocess=True, invert=False)
            logits_cpu = logits.detach().float().cpu()

            mean_logits = logits_cpu * w if mean_logits is None else mean_logits + logits_cpu * w

            m = m.to("cpu")
            torch.cuda.empty_cache()

        out = float(cast(torch.Tensor, postprocess_transforms(mean_logits)).item())
    
    logging.info(f"Ensemble prediction completed with probability {out}. "
                 f"Writing output to {args.output}...")

    # Save
    write_probability(args.output, out)
    logging.info(f"Output written to {args.output}.")
    logging.info(f"====BRAIN AGE PREDICTION PIPELINE COMPLETED====\n")

def main():
    """
    Main function to run the inference pipeline.
    """
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    predict_task_2()

if __name__ == "__main__":
    main()