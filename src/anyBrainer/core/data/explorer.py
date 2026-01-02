"""Fetches filenames to be processed from base directory.

Handles various dataset formats, including BIDS and a generic NIfTI one
(root->subject->session->files).
"""

__all__ = [
    "DataHandler",
    "GenericNiftiDataExplorer",
]

import logging
from typing import Iterator, Sequence, Literal
from pathlib import Path
from tqdm import tqdm

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.utils import resolve_path
from anyBrainer.interfaces import DataExplorer

logger = logging.getLogger(__name__)


class DataHandler:
    """General handler for MRI data, supporting multiple dataset formats.

    Supported: BIDS, GenericNifti
    """

    def __init__(
        self,
        data_dir: Path | str,
        data_format: Literal["GenericNifti", "BIDS"] = "GenericNifti",
        files_dir: str | None = None,
        files_suffix: str | None = None,
        exts: Sequence[str] = (".nii.gz", ".nii"),
    ):
        self.base_dir = resolve_path(data_dir)
        self.format = data_format
        self.files_dir = files_dir
        self.files_suffix = files_suffix
        self.exts = exts

        self.explorer = self._init_explorer()

    def _init_explorer(self):
        if self.format == "GenericNifti":
            return GenericNiftiDataExplorer(base_dir=self.base_dir)
        else:
            logger.error("Invalid data format.")
            raise ValueError(f"Unsupported data format: {self.format}")

    def get_all_nifti_files(self, as_list: bool = True) -> list[Path] | Iterator[Path]:
        return self.explorer.get_all_image_files(
            files_suffix=self.files_suffix,
            files_dir=self.files_dir,
            exts=self.exts,
            as_list=as_list,
        )

    def get_path(self) -> Path:
        return self.base_dir


@register(RK.DATA_EXPLORER)
class GenericNiftiDataExplorer(DataExplorer):
    """Handles generic datasets with subject/session format:

    root_dir/   └── subject_01/         └── session_01/ (optional) └──
    image.nii.gz
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def get_subject_dirs(self) -> list[Path]:
        return [d for d in self.base_dir.iterdir() if d.is_dir()]

    def get_session_dirs(self, subject_dir: Path) -> list[Path]:
        entries = [d for d in subject_dir.iterdir() if d.is_dir()]
        return entries if entries else [subject_dir]

    def get_image_files(
        self,
        session_dir: Path,
        files_suffix: str | None = None,
        exts: Sequence[str] = (".nii.gz", ".nii"),
        **kwargs,
    ) -> list[Path]:
        files = []
        for f in session_dir.iterdir():
            if f.is_file():
                if files_suffix is None:
                    if any(f.name.endswith(ext) for ext in exts):
                        files.append(f)
                else:
                    if any(f.name.endswith(f"{files_suffix}{ext}") for ext in exts):
                        files.append(f)
        return files

    def _iter_image_files(self, files_suffix, exts):
        for subject in tqdm(self.get_subject_dirs(), desc="Retrieving subjects"):
            for session in self.get_session_dirs(subject):
                for f in self.get_image_files(session, files_suffix, exts):
                    yield f

    def get_all_image_files(
        self,
        files_suffix: str | None = None,
        exts: Sequence[str] = (".nii.gz", ".nii"),
        as_list: bool = True,
        **kwargs,
    ) -> list[Path] | Iterator[Path]:
        it = self._iter_image_files(files_suffix, exts)
        return list(it) if as_list else it
