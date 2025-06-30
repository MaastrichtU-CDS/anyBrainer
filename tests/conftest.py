import sys
from pathlib import Path
import logging

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    LoadImage,
)
from monai.data import create_test_image_3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("anyBrainer")

if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


@pytest.fixture(scope="session")
def sample_data():
    img, seg = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "img": torch.tensor(img), 
        "img_1": torch.tensor(img),
        "brain_mask": torch.tensor(seg).long(),
        "sub_id": "1",
        "ses_id": "1",
        "modality": "t1",
        "count": 2,
    }

@pytest.fixture(scope="session")
def sample_data_contrastive():
    img, _ = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "query": torch.tensor(img), 
        "key": torch.tensor(img),
        "sub_id": "1",
        "ses_id": "1",
        "modality": "t1",
        "count": 2,
    }

