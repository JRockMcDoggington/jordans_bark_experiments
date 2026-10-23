#!/usr/bin/env python3
"""EfficientNet resize/learning-rate experiments with subject-level cross-validation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.transforms import v2


DEFAULT_DATA_FOLDER = Path("image_dir")
DEFAULT_CSV_NAME = "labels.csv"
DEFAULT_PHOTOS_NAME = "images"
PHOTO_COLUMNS = ("Photo0", "Photo1", "Photo2", "Photo3")
RESIZE_STRATEGIES = (
    "resize",
    "big_center_crop_resize",
    "small_center_crop_resize",
)
LEARNING_RATES = (0.001, 0.0001, 0.0003)
OUTER_FOLDS = 5
INNER_FOLDS = 8
MIN_TRAIN_TREES = 7
MIN_VALIDATION_TREES = 1
MIN_TEST_TREES = 2
MIN_TOTAL_TREES = MIN_TRAIN_TREES + MIN_VALIDATION_TREES + MIN_TEST_TREES
CHECKPOINT_TOLERANCE = 1e-12
ASSIGNMENT_VERSION = "resize_lr_tree_folds_v1"
MIXED_PRECISION_TRAINING = False
TRAINING_PRECISION = "float32"
BATCH_FALLBACK_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
)
OUTER_TEST_DESCRIPTION = (
    "Outer cross-validation test subset used to compare resize/LR experiment configurations; "
    "this is not the separate quarantined final holdout."
)
FINAL_HOLDOUT_COMPONENT_PREFIXES = (
    "quarantined",
)
REQUIRED_RAW_ARRAY_NAMES = (
    "logits",
    "probabilities",
    "true_labels",
    "predicted_labels",
)


@dataclass(frozen=True)
class ModelSpec:
    constructor: Callable[..., nn.Module]
    weights_enum: Any
    micro_batch_size: int
    grad_accum_steps: int
    eval_batch_size: int

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    def with_batch_configuration(
        self, micro_batch_size: int, grad_accum_steps: int
    ) -> "ModelSpec":
        return ModelSpec(
            self.constructor,
            self.weights_enum,
            micro_batch_size,
            grad_accum_steps,
            micro_batch_size,
        )


@dataclass(frozen=True)
class ArtifactDirectories:
    """Configuration-specific locations below the three requested artifact roots."""

    results: Path
    outputs: Path
    checkpoints: Path


# Explicit constructors and weight enums are required for torchvision 0.16.2.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "efficientnet_b0": ModelSpec(efficientnet_b0, EfficientNet_B0_Weights, 64, 1, 64),
    "efficientnet_b1": ModelSpec(efficientnet_b1, EfficientNet_B1_Weights, 64, 1, 64),
    "efficientnet_b2": ModelSpec(efficientnet_b2, EfficientNet_B2_Weights, 64, 1, 64),
    "efficientnet_b3": ModelSpec(efficientnet_b3, EfficientNet_B3_Weights, 64, 1, 64),
    "efficientnet_b4": ModelSpec(efficientnet_b4, EfficientNet_B4_Weights, 64, 1, 64),
    "efficientnet_b5": ModelSpec(efficientnet_b5, EfficientNet_B5_Weights, 64, 1, 64),
    "efficientnet_b6": ModelSpec(efficientnet_b6, EfficientNet_B6_Weights, 64, 1, 64),
    "efficientnet_b7": ModelSpec(efficientnet_b7, EfficientNet_B7_Weights, 64, 1, 64),
    "efficientnet_v2_s": ModelSpec(
        efficientnet_v2_s, EfficientNet_V2_S_Weights, 64, 1, 64
    ),
    "efficientnet_v2_m": ModelSpec(
        efficientnet_v2_m, EfficientNet_V2_M_Weights, 64, 1, 64
    ),
    "efficientnet_v2_l": ModelSpec(
        efficientnet_v2_l, EfficientNet_V2_L_Weights, 64, 1, 64
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one model/learning-rate/resize configuration through "
            "five subject-level cross-validation folds."
        )
    )
    parser.add_argument("--use-preassigned-folds", action="store_true")
    parser.add_argument("--data-folder", type=Path, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--photos-path", type=Path, default=None)
    parser.add_argument("--experiment-results-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--fold-assignments-path",
        type=Path,
        default=Path("fold_assignments.csv"),
    )
    parser.add_argument("--experiment-id", default="custom_run")
    parser.add_argument(
        "--model", choices=tuple(MODEL_REGISTRY), default="efficientnet_b0"
    )
    parser.add_argument(
        "--learning-rate", type=float, choices=LEARNING_RATES, default=0.001
    )
    parser.add_argument(
        "--resize-strategy", choices=RESIZE_STRATEGIES, default="resize"
    )
    parser.add_argument("--big-crop-size", type=int, default=960)
    parser.add_argument("--small-crop-size", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument(
        "--auto-batch-fallback",
        action="store_true",
        help=(
            "Start at 64 images and, only after a CUDA out-of-memory exception, restart "
            "the entire configuration at 32x2, 16x4, then 8x8."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Integer 0..4294967290. Default: reuse the saved run/fold seed, or choose a random seed for a new run.",
    )
    parser.add_argument("--folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument(
        "--require-cached-imagenet-weights",
        action="store_true",
        help=(
            "Refuse to download weights during training; the selected DEFAULT checkpoint "
            "must already exist below TORCH_HOME."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-data-only", action="store_true",
        help="Check labels, image files, crop dimensions, and fold feasibility on CPU, then exit.",
    )
    mode.add_argument(
        "--prepare-fold-assignments-only",
        action="store_true",
        help="Create the shared assignment file if absent, then validate it and exit.",
    )
    mode.add_argument(
        "--validate-fold-assignments-only",
        action="store_true",
        help="Validate the existing shared assignment file and exit.",
    )
    mode.add_argument(
        "--verify-transforms-only",
        action="store_true",
        help="Verify all model/strategy transform outputs without loading weights or data.",
    )
    mode.add_argument(
        "--verify-imagenet-weights-only",
        action="store_true",
        help=(
            "Resolve and load every registry model's DEFAULT ImageNet checkpoint in one "
            "process, report each result, and exit without reading the dataset or training."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> ModelSpec:
    if args.seed is not None and not 0 <= args.seed <= 4_294_967_290:
        raise ValueError("--seed must be between 0 and 4294967290 inclusive (five fold offsets are added).")
    if not args.experiment_id or not args.experiment_id.strip():
        raise ValueError("--experiment-id must not be empty.")
    if any(ch in args.experiment_id for ch in ("/", "\\", "\0")):
        raise ValueError("--experiment-id must not contain path separators or NUL bytes.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.folds != OUTER_FOLDS:
        raise ValueError(f"resize/LR experiment requires --folds {OUTER_FOLDS}; received {args.folds}.")
    if args.inner_folds != INNER_FOLDS:
        raise ValueError(
            f"The selected tree-level split requires --inner-folds {INNER_FOLDS}; "
            f"received {args.inner_folds}."
        )
    if args.limit_folds is not None and not 1 <= args.limit_folds <= args.folds:
        raise ValueError(f"--limit-folds must be between 1 and {args.folds}.")
    if args.big_crop_size != 960:
        raise ValueError("resize/LR experiment requires --big-crop-size 960 exactly.")
    if args.small_crop_size != 480:
        raise ValueError("resize/LR experiment requires --small-crop-size 480 exactly.")

    registry_spec = MODEL_REGISTRY[args.model]
    if args.batch_size is None:
        args.batch_size = registry_spec.micro_batch_size
    if args.grad_accum_steps is None:
        args.grad_accum_steps = registry_spec.grad_accum_steps
    requested_batch_configuration = (args.batch_size, args.grad_accum_steps)
    if requested_batch_configuration not in BATCH_FALLBACK_SCHEDULE:
        raise ValueError(
            "--batch-size and --grad-accum-steps must be one of: "
            + ", ".join(
                f"{micro}x{accumulation}"
                for micro, accumulation in BATCH_FALLBACK_SCHEDULE
            )
            + f"; received {args.batch_size}x{args.grad_accum_steps}."
        )
    if args.batch_size * args.grad_accum_steps != 64:
        raise ValueError("Every resize/LR experiment model must have effective training batch size 64.")
    if args.auto_batch_fallback and requested_batch_configuration != BATCH_FALLBACK_SCHEDULE[0]:
        raise ValueError(
            "--auto-batch-fallback must start at --batch-size 64 and "
            "--grad-accum-steps 1."
        )
    roots = [
        args.experiment_results_dir.expanduser().resolve(),
        args.checkpoints_dir.expanduser().resolve(),
    ]
    if len(set(roots)) != len(roots):
        raise ValueError(
            "The results, outputs, and checkpoint directories must not overlap."
        )
    if (
        roots[0] == roots[1]
        or roots[0].is_relative_to(roots[1])
        or roots[1].is_relative_to(roots[0])
    ):
        raise ValueError(
            "The results, outputs, and checkpoint directories must not overlap."
        )
    if registry_spec.effective_batch_size != 64:
        raise ValueError(f"Invalid registry entry for {args.model}: effective batch is not 64.")
    return registry_spec.with_batch_configuration(*requested_batch_configuration)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def reject_final_holdout_path(path: Path, argument_name: str) -> None:
    for component in path.parts:
        normalized = component.casefold().replace("-", "_")
        if normalized.startswith(FINAL_HOLDOUT_COMPONENT_PREFIXES):
            raise ValueError(
                f"{argument_name} points to the quarantined final holdout set: {path}. "
                "Use only the development dataset for model selection; the "
                "quarantined final holdout is excluded from all resize/LR experiment operations."
            )


def resolve_data_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    data_folder = args.data_folder.expanduser().resolve()
    csv_path = (
        args.csv_path.expanduser().resolve()
        if args.csv_path is not None
        else data_folder / DEFAULT_CSV_NAME
    )
    photos_path = (
        args.photos_path.expanduser().resolve()
        if args.photos_path is not None
        else data_folder / DEFAULT_PHOTOS_NAME
    )
    reject_final_holdout_path(data_folder, "--data-folder")
    reject_final_holdout_path(csv_path, "--csv-path")
    reject_final_holdout_path(photos_path, "--photos-path")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    if not photos_path.is_dir():
        raise FileNotFoundError(f"Photo directory does not exist: {photos_path}")
    return data_folder, csv_path, photos_path


def load_tree_dataframe(csv_path: Path) -> Tuple[pd.DataFrame, LabelEncoder]:
    try:
        tree_df = pd.read_csv(csv_path, dtype={"OBJECTID": "string", "Species": "string"}, keep_default_na=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read dataset CSV: {csv_path}") from exc

    required = {"OBJECTID", "Species", *PHOTO_COLUMNS}
    missing_columns = sorted(required.difference(tree_df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset CSV is missing required columns: {', '.join(missing_columns)}"
        )
    if tree_df.empty:
        raise ValueError(f"Dataset CSV has no tree rows: {csv_path}")

    tree_df = tree_df.loc[:, ["OBJECTID", "Species", *PHOTO_COLUMNS]].copy()
    tree_df["OBJECTID"] = tree_df["OBJECTID"].astype("string").str.strip()
    tree_df["Species"] = tree_df["Species"].astype("string").str.strip()
    invalid_ids = tree_df["OBJECTID"].isna() | (tree_df["OBJECTID"] == "")
    invalid_species = tree_df["Species"].isna() | (tree_df["Species"] == "")
    if invalid_ids.any():
        raise ValueError(f"Dataset contains {int(invalid_ids.sum())} empty OBJECTID values.")
    if invalid_species.any():
        raise ValueError(f"Dataset contains {int(invalid_species.sum())} empty Species values.")
    duplicate_ids = tree_df.loc[
        tree_df["OBJECTID"].duplicated(keep=False), "OBJECTID"
    ].unique()
    if len(duplicate_ids):
        preview = ", ".join(map(str, duplicate_ids[:20]))
        raise ValueError(
            "The tree-level CSV must contain exactly one row per OBJECTID. "
            f"Duplicate OBJECTIDs include: {preview}"
        )

    for column in PHOTO_COLUMNS:
        tree_df[column] = tree_df[column].fillna("").astype(str).str.strip()
    no_images = (tree_df.loc[:, list(PHOTO_COLUMNS)] == "").all(axis=1)
    if no_images.any():
        preview = ", ".join(tree_df.loc[no_images, "OBJECTID"].head(20).tolist())
        raise ValueError(f"Trees with no usable Photo0-Photo3 entries: {preview}")

    tree_df = tree_df.sort_values(
        ["Species", "OBJECTID"], kind="mergesort"
    ).reset_index(drop=True)
    species_counts = tree_df["Species"].value_counts().sort_index()
    if len(species_counts) < 2:
        raise ValueError("Classification requires at least two distinct classes in Species.")
    insufficient = species_counts[species_counts < MIN_TOTAL_TREES]
    if not insufficient.empty:
        details = "; ".join(f"{name}={count}" for name, count in insufficient.items())
        raise ValueError(
            "Every class needs at least 10 independent subjects to guarantee 7 training, "
            f"1 validation, and 2 test subjects per fold. Insufficient classes: {details}"
        )

    encoder = LabelEncoder()
    tree_df["Label"] = encoder.fit_transform(tree_df["Species"])
    return tree_df, encoder




def melt_and_validate_images(tree_df: pd.DataFrame, photos_path: Path) -> pd.DataFrame:
    image_df = tree_df.melt(
        id_vars=["OBJECTID", "Species", "Label"],
        value_vars=list(PHOTO_COLUMNS),
        var_name="Photo",
        value_name="ImageFilename",
    )
    image_df = image_df[image_df["ImageFilename"] != ""].copy()
    if image_df.empty:
        raise ValueError("No usable image paths remain after melting Photo0-Photo3.")

    resolved_paths: List[str] = []
    for filename in image_df["ImageFilename"].tolist():
        relative = Path(filename)
        if relative.is_absolute():
            raise ValueError(
                f"Expected a filename relative to --photos-path, found absolute path: {filename}"
            )
        resolved = (photos_path / relative).resolve()
        if not resolved.is_relative_to(photos_path.resolve()):
            raise ValueError(f"Image path escapes --photos-path: {filename}")
        resolved_paths.append(str(resolved))
    image_df["ImagePath"] = resolved_paths
    repeated = image_df.loc[image_df["ImagePath"].duplicated(keep=False), "ImageFilename"]
    if not repeated.empty:
        raise ValueError(
            "Each image must be listed only once; duplicate paths can leak images across folds: "
            + ", ".join(repeated.head(20))
        )

    missing = sorted(
        path for path in image_df["ImagePath"].unique() if not Path(path).is_file()
    )
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} referenced image files do not exist. First missing paths:\n{preview}"
        )
    image_df = image_df.sort_values(
        ["Species", "OBJECTID", "Photo", "ImageFilename"], kind="mergesort"
    ).reset_index(drop=True)
    return image_df


def generate_tree_assignments(
    tree_df: pd.DataFrame,
    seed: int,
    folds: int,
    inner_folds: int,
) -> pd.DataFrame:
    assignments = tree_df.loc[:, ["OBJECTID", "Species", "Label"]].copy()
    assignments = assignments.set_index("OBJECTID", drop=False)
    fold_columns = [f"fold_{fold}_subset" for fold in range(1, folds + 1)]
    for column in fold_columns:
        assignments[column] = ""
    assignments["outer_test_fold"] = 0

    outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(tree_df, tree_df["Species"]), start=1
    ):
        train_val_trees = tree_df.iloc[train_val_idx].reset_index(drop=True)
        test_ids = tree_df.iloc[test_idx]["OBJECTID"].tolist()

        # Eight inner folds select approximately one eighth of the outer train/val
        # pool for validation while guaranteeing one validation tree per species.
        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + fold
        )
        train_idx, validation_idx = next(
            inner_cv.split(train_val_trees, train_val_trees["Species"])
        )
        train_ids = train_val_trees.iloc[train_idx]["OBJECTID"].tolist()
        validation_ids = train_val_trees.iloc[validation_idx]["OBJECTID"].tolist()

        column = f"fold_{fold}_subset"
        assignments.loc[train_ids, column] = "train"
        assignments.loc[validation_ids, column] = "validation"
        assignments.loc[test_ids, column] = "test"
        assignments.loc[test_ids, "outer_test_fold"] = fold

    assignments = assignments.reset_index(drop=True)
    assignments.insert(0, "assignment_version", ASSIGNMENT_VERSION)
    assignments.insert(1, "seed", seed)
    assignments.insert(2, "folds", folds)
    assignments.insert(3, "inner_folds", inner_folds)
    return assignments


def validate_tree_assignments(
    assignments: pd.DataFrame,
    tree_df: pd.DataFrame,
    seed: int,
    folds: int,
    inner_folds: int,
) -> Dict[str, Any]:
    fold_columns = [f"fold_{fold}_subset" for fold in range(1, folds + 1)]
    required = {
        "assignment_version",
        "seed",
        "folds",
        "inner_folds",
        "OBJECTID",
        "Species",
        "Label",
        "outer_test_fold",
        *fold_columns,
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Fold-assignment file is missing columns: {', '.join(missing)}")
    if len(assignments) != len(tree_df):
        raise ValueError(
            f"Fold assignments contain {len(assignments)} trees, expected {len(tree_df)}."
        )

    assignments = assignments.copy()
    assignments["OBJECTID"] = assignments["OBJECTID"].astype("string").str.strip()
    if assignments["OBJECTID"].duplicated().any():
        duplicates = assignments.loc[
            assignments["OBJECTID"].duplicated(keep=False), "OBJECTID"
        ].unique()
        raise ValueError(
            "Fold assignments contain duplicate OBJECTIDs: "
            + ", ".join(map(str, duplicates[:20]))
        )

    expected_meta = {
        "assignment_version": ASSIGNMENT_VERSION,
        "seed": seed,
        "folds": folds,
        "inner_folds": inner_folds,
    }
    for column, expected in expected_meta.items():
        unique_values = assignments[column].drop_duplicates().tolist()
        if len(unique_values) != 1 or str(unique_values[0]) != str(expected):
            raise ValueError(
                f"Fold assignments have inconsistent {column}: {unique_values}; "
                f"expected {expected}."
            )

    expected_trees = tree_df.loc[:, ["OBJECTID", "Species", "Label"]].copy()
    expected_trees["OBJECTID"] = expected_trees["OBJECTID"].astype("string")
    comparison = expected_trees.merge(
        assignments.loc[:, ["OBJECTID", "Species", "Label"]],
        on="OBJECTID",
        how="outer",
        suffixes=("_expected", "_saved"),
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise ValueError("Fold assignments do not contain the exact eligible OBJECTID set.")
    species_match = comparison["Species_expected"].astype(str) == comparison[
        "Species_saved"
    ].astype(str)
    label_match = comparison["Label_expected"].astype(int) == comparison[
        "Label_saved"
    ].astype(int)
    if not species_match.all() or not label_match.all():
        bad = comparison.loc[~(species_match & label_match), "OBJECTID"].head(20)
        raise ValueError(
            "Species or encoded labels disagree between the dataset and assignments for: "
            + ", ".join(bad.astype(str))
        )

    allowed_subsets = {"train", "validation", "test"}
    test_counts_by_tree = np.zeros(len(assignments), dtype=np.int64)
    fold_summaries: List[Dict[str, Any]] = []
    species_order = sorted(tree_df["Species"].unique())
    for fold, column in enumerate(fold_columns, start=1):
        invalid = ~assignments[column].isin(allowed_subsets)
        if invalid.any():
            examples = assignments.loc[invalid, ["OBJECTID", column]].head(20)
            raise ValueError(
                f"Fold {fold} contains invalid or empty subset values:\n"
                + examples.to_string(index=False)
            )
        test_mask = assignments[column] == "test"
        test_counts_by_tree += test_mask.to_numpy(dtype=np.int64)
        saved_outer = assignments.loc[test_mask, "outer_test_fold"].astype(int)
        if not (saved_outer == fold).all():
            raise ValueError(f"outer_test_fold disagrees with {column}.")

        violations: List[str] = []
        minimum_counts = {
            "train": MIN_TRAIN_TREES,
            "validation": MIN_VALIDATION_TREES,
            "test": MIN_TEST_TREES,
        }
        per_species_minima: Dict[str, int] = {}
        for subset, minimum in minimum_counts.items():
            subset_counts = (
                assignments.loc[assignments[column] == subset, "Species"]
                .value_counts()
                .reindex(species_order, fill_value=0)
            )
            per_species_minima[subset] = int(subset_counts.min())
            for species, count in subset_counts.items():
                if int(count) < minimum:
                    all_counts = {
                        name: int(
                            (
                                assignments.loc[assignments[column] == name, "Species"]
                                == species
                            ).sum()
                        )
                        for name in ("train", "validation", "test")
                    }
                    violations.append(
                        f"fold={fold}, species={species}, train={all_counts['train']}, "
                        f"validation={all_counts['validation']}, test={all_counts['test']}, "
                        "required=train>=7/validation>=1/test>=2"
                    )
        if violations:
            raise ValueError("Per-species tree minimum violation(s):\n" + "\n".join(violations))

        fold_summaries.append(
            {
                "fold": fold,
                "train_trees": int((assignments[column] == "train").sum()),
                "validation_trees": int((assignments[column] == "validation").sum()),
                "test_trees": int(test_mask.sum()),
                "minimum_train_trees_per_species": per_species_minima["train"],
                "minimum_validation_trees_per_species": per_species_minima["validation"],
                "minimum_test_trees_per_species": per_species_minima["test"],
            }
        )

    if not np.all(test_counts_by_tree == 1):
        bad_rows = assignments.loc[test_counts_by_tree != 1, "OBJECTID"].head(20)
        raise ValueError(
            "Every eligible tree must appear in exactly one outer test fold. Violations: "
            + ", ".join(bad_rows.astype(str))
        )
    saved_outer_folds = assignments["outer_test_fold"].astype(int)
    if not saved_outer_folds.between(1, folds).all():
        raise ValueError("outer_test_fold must be between 1 and 5 for every tree.")

    return {
        "assignment_version": ASSIGNMENT_VERSION,
        "tree_count": int(len(assignments)),
        "species_count": int(tree_df["Species"].nunique()),
        "every_tree_tested_exactly_once": True,
        "folds": fold_summaries,
    }


def load_assignment_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared fold assignments do not exist: {path}\n"
            "Run with --prepare-fold-assignments-only before starting the array."
        )
    try:
        return pd.read_csv(
            path,
            dtype={
                "OBJECTID": "string",
                "Species": "string",
                "assignment_version": "string",
            },
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not read shared fold assignments: {path}") from exc


def prepare_or_validate_assignments(
    path: Path,
    tree_df: pd.DataFrame,
    seed: int,
    folds: int,
    inner_folds: int,
    create_if_missing: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(
                f"Shared fold assignments do not exist: {path}\n"
                "Run with --prepare-fold-assignments-only before starting the array."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        generated = generate_tree_assignments(tree_df, seed, folds, inner_folds)
        validate_tree_assignments(generated, tree_df, seed, folds, inner_folds)
        generated.to_csv(path, index=False, mode="x", lineterminator="\n")
        print(f"Created shared fold assignments: {path}")
    else:
        print(f"Reusing existing shared fold assignments: {path}")

    assignments = load_assignment_csv(path)
    summary = validate_tree_assignments(assignments, tree_df, seed, folds, inner_folds)
    summary["path"] = str(path)
    summary["identifier"] = ASSIGNMENT_VERSION
    return assignments, summary


def make_image_assignments(
    image_df: pd.DataFrame, tree_assignments: pd.DataFrame, folds: int
) -> pd.DataFrame:
    assignment_index = tree_assignments.set_index("OBJECTID")
    chunks: List[pd.DataFrame] = []
    for fold in range(1, folds + 1):
        chunk = image_df.copy()
        chunk.insert(0, "fold", fold)
        chunk.insert(
            1,
            "subset",
            chunk["OBJECTID"].map(assignment_index[f"fold_{fold}_subset"]),
        )
        if chunk["subset"].isna().any():
            raise ValueError(f"Fold {fold} has image rows without a tree assignment.")
        chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True)
    return result


def save_image_assignments(image_assignments: pd.DataFrame, path: Path) -> None:
    output = image_assignments.rename(
        columns={
            "Species": "species",
            "Label": "encoded_label",
            "ImagePath": "image_path",
            "ImageFilename": "image_filename",
            "Photo": "photo_column",
        }
    )
    columns = [
        "fold",
        "subset",
        "OBJECTID",
        "species",
        "encoded_label",
        "image_path",
        "image_filename",
        "photo_column",
    ]
    output.loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ImageClassificationDataset(Dataset):
    def __init__(self, image_df: pd.DataFrame, transform: Callable[[Image.Image], Any]):
        self.image_df = image_df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_df)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor, int]:
        row = self.image_df.iloc[index]
        image_path = str(row["ImagePath"])
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Error loading image at {image_path}") from exc
        image = self.transform(image)
        label = torch.tensor(int(row["Label"]), dtype=torch.long)
        return image, label, index


def get_dataloader(
    image_df: pd.DataFrame,
    transform: Callable[[Image.Image], Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    if image_df.empty:
        raise ValueError("Cannot create a DataLoader for an empty image subset.")
    dataset = ImageClassificationDataset(image_df, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def normalize_size(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    values = list(value)
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError(f"Unsupported torchvision preprocessing size: {value}")


def preprocessing_info(spec: ModelSpec) -> Tuple[Any, Dict[str, Any]]:
    weights = spec.weights_enum.DEFAULT
    preset = weights.transforms()
    final_size = normalize_size(preset.crop_size)
    resize_size = normalize_size(preset.resize_size)
    antialias = getattr(preset, "antialias", None)
    info = {
        "weight_enum": f"{spec.weights_enum.__name__}.DEFAULT",
        "resolved_weight": str(weights),
        "default_resize_size": list(resize_size),
        "final_input_dimensions": list(final_size),
        "normalization_mean": [float(value) for value in preset.mean],
        "normalization_std": [float(value) for value in preset.std],
        "interpolation_mode": str(preset.interpolation),
        "antialias": antialias,
    }
    return preset, info


def crop_size_for_strategy(args: argparse.Namespace) -> Optional[int]:
    if args.resize_strategy == "resize":
        return None
    if args.resize_strategy == "big_center_crop_resize":
        return args.big_crop_size
    if args.resize_strategy == "small_center_crop_resize":
        return args.small_crop_size
    raise ValueError(f"Unsupported resize strategy: {args.resize_strategy}")


def build_transform(
    spec: ModelSpec,
    resize_strategy: str,
    big_crop_size: int,
    small_crop_size: int,
) -> Tuple[Callable[[Image.Image], Any], Dict[str, Any]]:
    preset, info = preprocessing_info(spec)
    final_size = tuple(info["final_input_dimensions"])
    spatial_ops: List[Any] = [v2.ToImage()]
    crop_size: Optional[int] = None
    if resize_strategy == "big_center_crop_resize":
        crop_size = big_crop_size
        spatial_ops.append(v2.CenterCrop((crop_size, crop_size)))
    elif resize_strategy == "small_center_crop_resize":
        crop_size = small_crop_size
        spatial_ops.append(v2.CenterCrop((crop_size, crop_size)))
    elif resize_strategy != "resize":
        raise ValueError(f"Unsupported resize strategy: {resize_strategy}")

    resize_kwargs: Dict[str, Any] = {"interpolation": preset.interpolation}
    if hasattr(preset, "antialias"):
        resize_kwargs["antialias"] = preset.antialias
    spatial_ops.append(v2.Resize(final_size, **resize_kwargs))
    spatial_ops.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=preset.mean, std=preset.std),
        ]
    )
    info["resize_strategy"] = resize_strategy
    info["resize_lr_crop_size_pixels"] = crop_size
    info["spatial_pipeline"] = (
        "direct_resize"
        if crop_size is None
        else f"exact_{crop_size}x{crop_size}_center_crop_then_resize"
    )
    return v2.Compose(spatial_ops), info


def verify_all_transforms() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    source = Image.new("RGB", (1280, 960), color=(127, 127, 127))
    for model_name, spec in MODEL_REGISTRY.items():
        preset, expected = preprocessing_info(spec)
        for strategy in RESIZE_STRATEGIES:
            transform, info = build_transform(spec, strategy, 960, 480)
            tensor = transform(source)
            actual_size = tuple(int(value) for value in tensor.shape[-2:])
            expected_size = tuple(expected["final_input_dimensions"])
            if actual_size != expected_size:
                raise AssertionError(
                    f"{model_name}/{strategy} produced {actual_size}, expected {expected_size}."
                )
            if info["normalization_mean"] != [float(value) for value in preset.mean]:
                raise AssertionError(f"Normalization mean mismatch for {model_name}.")
            if info["normalization_std"] != [float(value) for value in preset.std]:
                raise AssertionError(f"Normalization std mismatch for {model_name}.")
            if info["interpolation_mode"] != str(preset.interpolation):
                raise AssertionError(f"Interpolation mismatch for {model_name}.")
            results.append(
                {
                    "model": model_name,
                    "resize_strategy": strategy,
                    "output_dimensions": list(actual_size),
                    "weight_enum": info["weight_enum"],
                    "interpolation_mode": info["interpolation_mode"],
                    "antialias": info["antialias"],
                }
            )
    return results


def weight_cache_details(spec: ModelSpec) -> Tuple[Any, Path]:
    weights = spec.weights_enum.DEFAULT
    weight_filename = Path(urlparse(weights.url).path).name
    if not weight_filename:
        raise ValueError(
            f"{spec.weights_enum.__name__}.DEFAULT does not provide a checkpoint filename."
        )
    cached_path = Path(torch.hub.get_dir()) / "checkpoints" / weight_filename
    return weights, cached_path


def verify_all_imagenet_weights() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME')}", flush=True)
    print(f"Torch hub directory={torch.hub.get_dir()}", flush=True)
    for model_name, spec in MODEL_REGISTRY.items():
        weights, cached_path = weight_cache_details(spec)
        cached_before = cached_path.is_file()
        try:
            model = spec.constructor(weights=weights)
            if not cached_path.is_file():
                raise FileNotFoundError(
                    f"DEFAULT weights loaded but expected cache file is absent: {cached_path}"
                )
            result = {
                "model": model_name,
                "weight_enum": f"{spec.weights_enum.__name__}.DEFAULT",
                "resolved_weight": str(weights),
                "checkpoint": str(cached_path),
                "cached_before_check": bool(cached_before),
                "status": "success",
            }
            print(
                f"WEIGHT CHECK SUCCESS: {model_name}: {cached_path}",
                flush=True,
            )
            del model
        except Exception as exc:
            result = {
                "model": model_name,
                "weight_enum": f"{spec.weights_enum.__name__}.DEFAULT",
                "resolved_weight": str(weights),
                "checkpoint": str(cached_path),
                "cached_before_check": bool(cached_before),
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            print(f"WEIGHT CHECK FAILURE: {failures[-1]}", file=sys.stderr, flush=True)
        results.append(result)
        gc.collect()
    print(json.dumps({"weight_checks": results}, indent=2), flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(MODEL_REGISTRY)} ImageNet weight checks failed:\n"
            + "\n".join(failures)
        )
    return results


def preflight_crop_dimensions(
    image_df: pd.DataFrame, crop_size: Optional[int], args: argparse.Namespace
) -> None:
    if crop_size is None:
        return
    for image_path in sorted(image_df["ImagePath"].unique()):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError(f"Could not inspect image dimensions: {image_path}") from exc
        if width < crop_size or height < crop_size:
            raise ValueError(
                "Source image is too small for the requested center crop. "
                f"image_path={image_path!r}, original_dimensions={width}x{height}, "
                f"requested_crop={crop_size}x{crop_size}, "
                f"experiment_id={args.experiment_id!r}, model={args.model!r}, "
                f"learning_rate={args.learning_rate}, "
                f"resize_strategy={args.resize_strategy!r}. No padding was added."
            )


def create_model(
    spec: ModelSpec,
    num_classes: int,
    device: torch.device,
    require_cached_weights: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    weights, cached_path = weight_cache_details(spec)
    cached_before_load = cached_path.is_file()
    if require_cached_weights and not cached_before_load:
        raise FileNotFoundError(
            "The required pretrained checkpoint is not cached, and this training task is "
            "forbidden from downloading it. Run --verify-imagenet-weights-only once with "
            f"the same TORCH_HOME before submitting the batch jobs. Missing: {cached_path}"
        )
    try:
        if require_cached_weights:
            # Loading the file directly makes each training-job invocation network-free,
            # including the small time-of-check/time-of-use window after the
            # existence check above.
            cached_state_dict = torch.load(cached_path, map_location="cpu")
            model = spec.constructor(weights=None)
            model.load_state_dict(cached_state_dict, strict=True)
            del cached_state_dict
        else:
            model = spec.constructor(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the required pretrained ImageNet weights "
            f"{spec.weights_enum.__name__}.DEFAULT. Expected cache file: {cached_path}. "
            "The cached checkpoint was unavailable or unusable. "
            f"Original error: {exc}"
        ) from exc
    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Sequential):
        raise ValueError("Selected EfficientNet has an unsupported classifier structure.")
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise ValueError("Selected EfficientNet classifier does not end in nn.Linear.")
    model.classifier[-1] = nn.Linear(final_layer.in_features, num_classes)
    model = model.to(device)
    return model, {
        "torch_hub_directory": str(torch.hub.get_dir()),
        "expected_weight_cache_file": str(cached_path),
        "weight_was_cached_before_load": bool(cached_before_load),
        "weight_cache_file_exists_after_load": bool(cached_path.is_file()),
        "weight_load_mode": (
            "direct_cached_state_dict_no_network"
            if require_cached_weights
            else "torchvision_weights_enum"
        ),
    }


def ensure_finite_loss(loss: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"Nonfinite loss encountered during {context}: {loss.item()}")


def normalize_accumulated_gradients(
    optimizer: torch.optim.Optimizer, accumulated_sample_count: int
) -> None:
    if accumulated_sample_count <= 0:
        raise ValueError("An accumulation group must contain at least one sample.")
    denominator = float(accumulated_sample_count)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.div_(denominator)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    predictions: List[int] = []
    true_labels: List[int] = []
    optimizer.zero_grad(set_to_none=True)
    accumulation_micro_batches = 0
    accumulation_samples = 0

    for step, (images, labels, _indices) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        raw_loss = criterion(logits, labels)
        ensure_finite_loss(raw_loss, "training")

        # CrossEntropyLoss returns a micro-batch mean. Accumulate the corresponding
        # sample sums, then divide the gradients by the actual number of samples in
        # this optimization window. This matches a full 64-sample batch for a complete
        # accumulation group and also handles a final incomplete group or partial
        # DataLoader batch using the exact number of samples that actually contributed.
        micro_batch_samples = int(images.size(0))
        (raw_loss * micro_batch_samples).backward()
        accumulation_micro_batches += 1
        accumulation_samples += micro_batch_samples
        if accumulation_micro_batches == grad_accum_steps or step == len(dataloader):
            normalize_accumulated_gradients(optimizer, accumulation_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_micro_batches = 0
            accumulation_samples = 0

        running_loss += float(raw_loss.item()) * micro_batch_samples
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        true_labels.extend(labels.detach().cpu().tolist())

    if not true_labels:
        raise ValueError("Training DataLoader produced no samples.")
    return {
        "loss": running_loss / len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    capture_outputs: bool,
    context: str,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    predictions: List[int] = []
    true_labels: List[int] = []
    row_indices: List[int] = []
    logits_chunks: List[np.ndarray] = []

    for images, labels, indices in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(
                f"{context} logits have shape {tuple(logits.shape)}; expected "
                f"(samples, {num_classes})."
            )
        loss = criterion(logits, labels)
        ensure_finite_loss(loss, context)
        running_loss += float(loss.item()) * images.size(0)
        batch_predictions = logits.argmax(dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        true_labels.extend(labels.detach().cpu().tolist())
        row_indices.extend(indices.detach().cpu().tolist())
        if capture_outputs:
            # Keep this conversion explicit so saved outputs remain unambiguously float32.
            logits_cpu = logits.detach().float().cpu()
            logits_chunks.append(logits_cpu.numpy())

    if not true_labels:
        raise ValueError(f"{context} DataLoader produced no samples.")
    result: Dict[str, Any] = {
        "loss": running_loss / len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "true_labels": np.asarray(true_labels, dtype=np.int64),
        "predicted_labels": np.asarray(predictions, dtype=np.int64),
        "row_indices": np.asarray(row_indices, dtype=np.int64),
    }
    if capture_outputs:
        result["logits"] = np.concatenate(logits_chunks, axis=0)
    return result


def checkpoint_decision(
    val_macro_f1: float,
    val_loss: float,
    best_macro_f1: float,
    best_loss: float,
) -> Tuple[bool, bool, str]:
    if val_macro_f1 > best_macro_f1 + CHECKPOINT_TOLERANCE:
        return True, True, "selected: higher validation macro F1"
    tied_f1 = abs(val_macro_f1 - best_macro_f1) <= CHECKPOINT_TOLERANCE
    if tied_f1 and val_loss < best_loss - CHECKPOINT_TOLERANCE:
        return (
            True,
            False,
            "selected: validation macro F1 tied within 1e-12 and validation loss was lower",
        )
    if tied_f1 and abs(val_loss - best_loss) <= CHECKPOINT_TOLERANCE:
        return (
            False,
            False,
            "retained earlier checkpoint: validation macro F1 and loss tied within 1e-12",
        )
    return False, False, "retained checkpoint: validation macro F1 was lower"


def save_confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
    output_prefix: Path,
) -> Tuple[Path, Path]:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    matrix_df.to_csv(csv_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_size = max(10.0, len(class_names) * 0.35)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(output_prefix.name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return csv_path, png_path


def save_raw_outputs(
    result: Dict[str, Any],
    split_df: pd.DataFrame,
    outputs_dir: Path,
    fold: int,
    subset: str,
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    num_classes: int,
) -> Tuple[Path, Path]:
    logits = np.asarray(result["logits"], dtype=np.float32)
    true_labels = np.asarray(result["true_labels"], dtype=np.int64)
    row_indices = np.asarray(result["row_indices"], dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(
            f"Raw {subset} logits have shape {logits.shape}; expected (*, {num_classes})."
        )
    logits_cpu = torch.from_numpy(logits).float().cpu()
    probabilities_cpu = torch.softmax(logits_cpu, dim=1).cpu()
    probabilities = probabilities_cpu.numpy().astype(np.float32, copy=False)
    predicted_labels = logits.argmax(axis=1).astype(np.int64, copy=False)
    captured_predictions = np.asarray(result["predicted_labels"], dtype=np.int64)
    if not np.array_equal(predicted_labels, captured_predictions):
        raise ValueError(
            f"Raw {subset} predicted labels do not match the saved logits argmax."
        )

    sample_count = logits.shape[0]
    counts = {
        "logits": sample_count,
        "probabilities": probabilities.shape[0],
        "true_labels": true_labels.shape[0],
        "predicted_labels": predicted_labels.shape[0],
        "row_indices": row_indices.shape[0],
    }
    if len(set(counts.values())) != 1 or sample_count != len(split_df):
        raise ValueError(
            f"Raw {subset} output/metadata sample-count mismatch: {counts}, "
            f"metadata_source={len(split_df)}"
        )
    if sorted(row_indices.tolist()) != list(range(len(split_df))):
        raise ValueError(
            f"Raw {subset} output indices are missing, duplicated, or out of range."
        )
    metadata = split_df.iloc[row_indices].reset_index(drop=True)
    metadata_true = metadata["Label"].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_true, true_labels):
        raise ValueError(f"Raw {subset} labels do not align with metadata rows.")

    metadata_output = pd.DataFrame(
        {
            "row_index": np.arange(sample_count, dtype=np.int64),
            "fold": fold,
            "subset": subset,
            "OBJECTID": metadata["OBJECTID"].astype(str),
            "species": metadata["Species"].astype(str),
            "encoded_true_label": true_labels,
            "encoded_predicted_label": predicted_labels,
            "image_path": metadata["ImagePath"].astype(str),
            "image_filename": metadata["ImageFilename"].astype(str),
            "best_checkpoint_epoch": int(checkpoint["best_epoch"]),
            "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]),
            "model": args.model,
            "learning_rate": args.learning_rate,
            "resize_strategy": args.resize_strategy,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "evaluation_scope": (
                OUTER_TEST_DESCRIPTION
                if subset == "test"
                else "Inner validation subset used for checkpoint selection."
            ),
            "quarantined_final_holdout_used": False,
        }
    )
    if len(metadata_output) != sample_count:
        raise ValueError(f"Raw {subset} metadata row count changed unexpectedly.")
    if logits.dtype != np.float32:
        raise TypeError(f"Raw {subset} logits must be float32; found {logits.dtype}.")
    if probabilities.dtype != np.float32:
        raise TypeError(
            f"Raw {subset} probabilities must be float32; found {probabilities.dtype}."
        )
    if probabilities.shape != logits.shape:
        raise ValueError(
            f"Raw {subset} probabilities shape {probabilities.shape} does not match "
            f"logits shape {logits.shape}."
        )
    if true_labels.ndim != 1 or predicted_labels.ndim != 1:
        raise ValueError(f"Raw {subset} labels must be one-dimensional.")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError(f"Raw {subset} logits or probabilities are nonfinite.")
    probability_sums = probabilities.sum(axis=1, dtype=np.float32)
    if not np.allclose(probability_sums, 1.0, rtol=1e-5, atol=1e-6):
        maximum_error = float(np.max(np.abs(probability_sums - 1.0)))
        raise ValueError(
            f"Raw {subset} probability rows do not sum to 1 (max error={maximum_error})."
        )

    if subset == "validation":
        stem = f"fold_{fold}_best_validation"
    elif subset == "test":
        stem = f"fold_{fold}_test"
    else:
        raise ValueError(f"Unsupported raw-output subset: {subset}")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    npz_path = outputs_dir / f"{stem}_outputs.npz"
    metadata_path = outputs_dir / f"{stem}_metadata.csv"
    np.savez_compressed(
        npz_path,
        logits=logits,
        probabilities=probabilities,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )
    metadata_output.to_csv(metadata_path, index=False, lineterminator="\n")

    with np.load(npz_path) as saved:
        if set(saved.files) != set(REQUIRED_RAW_ARRAY_NAMES):
            raise ValueError(
                f"Saved {subset} NPZ contains unexpected arrays: {saved.files}."
            )
        saved_counts = {name: int(saved[name].shape[0]) for name in saved.files}
        if len(set(saved_counts.values())) != 1 or next(iter(saved_counts.values())) != len(
            metadata_output
        ):
            raise ValueError(
                f"Saved {subset} NPZ arrays do not align with metadata: {saved_counts}"
            )
        if saved["logits"].shape[1] != num_classes:
            raise ValueError(f"Saved {subset} logits have the wrong class dimension.")
        if saved["logits"].dtype != np.float32:
            raise TypeError(f"Saved {subset} logits are not float32.")
        if saved["probabilities"].dtype != np.float32:
            raise TypeError(f"Saved {subset} probabilities are not float32.")
    return npz_path, metadata_path


def configuration_artifact_directories(args: argparse.Namespace) -> ArtifactDirectories:
    """Return the stable model/configuration hierarchy without creating it."""
    root = args.experiment_results_dir.expanduser().resolve() / args.model / args.experiment_id

    return ArtifactDirectories(
        results=(
            root / "results"
        ),
        outputs=(
            root / "outputs"
        ),
        checkpoints=(
            args.checkpoints_dir.expanduser().resolve() / args.model / args.experiment_id
        ),
    )


def make_run_directories(
    args: argparse.Namespace,
) -> Tuple[ArtifactDirectories, str, bool]:
    directories = configuration_artifact_directories(args)
    targets = (directories.results, directories.outputs, directories.checkpoints)
    existing = [path.exists() for path in targets]
    created = not any(existing)
    if created:
        for path in targets:
            path.mkdir(parents=True, exist_ok=False)
        timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return directories, timestamp_token, True

    for path in targets:
        if not path.is_dir():
            raise NotADirectoryError(
                "All configuration artifact locations must already be directories when "
                f"resuming; invalid location: {path}"
            )
    config_path = directories.results / "run_config.json"
    if not config_path.is_file():
        entries = {
            category: sorted(path.name for path in directory.iterdir())
            for category, directory in (
                ("results", directories.results),
                ("outputs", directories.outputs),
                ("checkpoints", directories.checkpoints),
            )
        }
        raise RuntimeError(
            "The stable configuration directories already exist without a readable "
            f"run_config.json and cannot be identified safely: {entries}"
        )
    existing_config = load_json(config_path)
    timestamp_token = str(existing_config.get("timestamp_token", "unknown"))
    return directories, timestamp_token, False


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def slurm_metadata() -> Dict[str, Optional[str]]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "hostname": socket.gethostname(),
    }


def command_line_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: json_ready(value) for name, value in vars(args).items()}


def fold_split(
    image_assignments: pd.DataFrame, fold: int, subset: str
) -> pd.DataFrame:
    result = image_assignments[
        (image_assignments["fold"] == fold)
        & (image_assignments["subset"] == subset)
    ].copy()
    if result.empty:
        raise ValueError(f"Fold {fold} {subset} image subset is empty.")
    return result.reset_index(drop=True)


def configuration_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None or bool(pd.isna(actual))
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-15))
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_run_config_compatibility(
    config: Dict[str, Any],
    args: argparse.Namespace,
    spec: ModelSpec,
    crop_size: Optional[int],
    num_classes: int,
) -> None:
    expected = {
        "seed": args.seed,
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "crop_size_pixels": crop_size,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "outer_folds": args.folds,
        "inner_folds": args.inner_folds,
        "folds_requested": args.limit_folds or args.folds,
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation_steps": spec.grad_accum_steps,
        "effective_batch_size": spec.effective_batch_size,
        "mixed_precision_training": MIXED_PRECISION_TRAINING,
        "training_precision": TRAINING_PRECISION,
        "num_classes": num_classes,
    }
    mismatches = [
        f"{name}: saved={config.get(name)!r}, requested={value!r}"
        for name, value in expected.items()
        if not configuration_value_matches(config.get(name), value)
    ]
    if mismatches:
        raise ValueError(
            "The stable run directory belongs to a different or incompatible "
            "configuration and will not be overwritten:\n" + "\n".join(mismatches)
        )


def fold_artifact_paths(
    directories: ArtifactDirectories, fold: int
) -> Dict[str, Path]:
    return {
        "checkpoint": directories.checkpoints / f"fold_{fold}_best.pt",
        "epoch_metrics": directories.results / f"fold_{fold}_epoch_metrics.csv",
        "validation_outputs": directories.outputs / f"fold_{fold}_best_validation_outputs.npz",
        "validation_metadata": directories.outputs / f"fold_{fold}_best_validation_metadata.csv",
        "test_outputs": directories.outputs / f"fold_{fold}_test_outputs.npz",
        "test_metadata": directories.outputs / f"fold_{fold}_test_metadata.csv",
        "validation_metrics": directories.results / f"fold_{fold}_validation_metrics.json",
        "test_metrics": directories.results / f"fold_{fold}_test_metrics.json",
        "validation_confusion_csv": (
            directories.results / f"fold_{fold}_best_validation_confusion_matrix.csv"
        ),
        "validation_confusion_png": (
            directories.results / f"fold_{fold}_best_validation_confusion_matrix.png"
        ),
        "test_confusion_csv": directories.results / f"fold_{fold}_test_confusion_matrix.csv",
        "test_confusion_png": directories.results / f"fold_{fold}_test_confusion_matrix.png",
        "fold_result": directories.results / f"fold_{fold}_result.json",
    }


def clean_interrupted_fold_artifacts(
    directories: ArtifactDirectories, fold: int
) -> List[str]:
    allowed_roots = {
        directories.results.resolve(),
        directories.outputs.resolve(),
        directories.checkpoints.resolve(),
    }
    removed: List[str] = []
    for path in fold_artifact_paths(directories, fold).values():
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise RuntimeError(
                "Refusing to remove a fold artifact outside its configuration "
                f"directories: {resolved}"
            )
        if path.is_file():
            path.unlink()
            removed.append(str(path))
        elif path.exists():
            raise RuntimeError(f"Expected fold artifact to be a file: {path}")
    return removed


def validate_checkpoint_artifact(
    path: Path,
    fold: int,
    args: argparse.Namespace,
    spec: ModelSpec,
    crop_size: Optional[int],
    class_names: Sequence[str],
    expected_validation_samples: int,
    expected_test_samples: int,
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Fold {fold} checkpoint cannot be read: {path}") from exc
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("model_state_dict"), dict)
        or not checkpoint["model_state_dict"]
    ):
        raise ValueError(f"Fold {fold} checkpoint is not a valid checkpoint mapping.")
    expected = {
        "fold": fold,
        "experiment_id": args.experiment_id,
        "model_name": args.model,
        "imagenet_weight_enum": f"{spec.weights_enum.__name__}.DEFAULT",
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "crop_size_pixels": crop_size,
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation_steps": spec.grad_accum_steps,
        "effective_batch_size": spec.effective_batch_size,
        "mixed_precision_training": MIXED_PRECISION_TRAINING,
        "training_precision": TRAINING_PRECISION,
        "class_name_order": list(class_names),
        "num_classes": len(class_names),
        "expected_validation_samples": expected_validation_samples,
        "expected_test_samples": expected_test_samples,
    }
    mismatches = [
        f"{name}: saved={checkpoint.get(name)!r}, expected={value!r}"
        for name, value in expected.items()
        if not configuration_value_matches(checkpoint.get(name), value)
    ]
    if mismatches:
        raise ValueError(
            f"Fold {fold} checkpoint configuration mismatch:\n" + "\n".join(mismatches)
        )
    saved_cli = checkpoint.get("command_line_args")
    if not isinstance(saved_cli, dict):
        raise ValueError(f"Fold {fold} checkpoint lacks its command-line configuration.")
    cli_expected = {
        "epochs": args.epochs,
        "patience": args.patience,
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "limit_folds": args.limit_folds,
    }
    cli_mismatches = [
        f"{name}: saved={saved_cli.get(name)!r}, expected={value!r}"
        for name, value in cli_expected.items()
        if not configuration_value_matches(saved_cli.get(name), value)
    ]
    if cli_mismatches:
        raise ValueError(
            f"Fold {fold} checkpoint command configuration mismatch:\n"
            + "\n".join(cli_mismatches)
        )
    if int(checkpoint.get("best_epoch", 0)) <= 0:
        raise ValueError(f"Fold {fold} checkpoint has an invalid best_epoch.")
    return checkpoint


def validate_epoch_metrics_artifact(
    path: Path,
    fold: int,
    fold_result: Dict[str, Any],
    args: argparse.Namespace,
    spec: ModelSpec,
    crop_size: Optional[int],
) -> None:
    try:
        metrics = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Fold {fold} per-epoch metrics cannot be read: {path}") from exc
    required = {
        "fold",
        "epoch",
        "learning_rate",
        "model",
        "resize_strategy",
        "crop_size_pixels",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "mixed_precision_training",
        "training_precision",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing or metrics.empty:
        raise ValueError(
            f"Fold {fold} per-epoch metrics are empty or missing columns: {missing}"
        )
    expected_epochs = list(range(1, len(metrics) + 1))
    if metrics["epoch"].astype(int).tolist() != expected_epochs:
        raise ValueError(f"Fold {fold} per-epoch metric epochs are not contiguous from 1.")
    if len(metrics) != int(fold_result.get("epochs_completed", -1)):
        raise ValueError(f"Fold {fold} per-epoch metric count disagrees with fold result.")
    row_expectations = {
        "fold": fold,
        "learning_rate": args.learning_rate,
        "model": args.model,
        "resize_strategy": args.resize_strategy,
        "crop_size_pixels": crop_size,
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation_steps": spec.grad_accum_steps,
        "effective_batch_size": spec.effective_batch_size,
        "mixed_precision_training": MIXED_PRECISION_TRAINING,
        "training_precision": TRAINING_PRECISION,
    }
    for column, expected in row_expectations.items():
        if not all(configuration_value_matches(value, expected) for value in metrics[column]):
            raise ValueError(
                f"Fold {fold} per-epoch metrics contain incompatible {column} values."
            )


def validate_raw_output_artifacts(
    npz_path: Path,
    metadata_path: Path,
    expected_df: pd.DataFrame,
    fold: int,
    subset: str,
    args: argparse.Namespace,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    try:
        with np.load(npz_path) as saved:
            if set(saved.files) != set(REQUIRED_RAW_ARRAY_NAMES):
                raise ValueError(f"unexpected NPZ arrays: {saved.files}")
            arrays = {name: np.array(saved[name], copy=True) for name in saved.files}
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold} {subset} raw-output NPZ cannot be validated: {npz_path}"
        ) from exc
    logits = arrays["logits"]
    probabilities = arrays["probabilities"]
    true_labels = arrays["true_labels"]
    predicted_labels = arrays["predicted_labels"]
    expected_count = len(expected_df)
    if logits.dtype != np.float32 or probabilities.dtype != np.float32:
        raise TypeError(
            f"Fold {fold} {subset} logits/probabilities must both be float32."
        )
    if logits.shape != (expected_count, num_classes):
        raise ValueError(
            f"Fold {fold} {subset} logits shape is {logits.shape}; "
            f"expected {(expected_count, num_classes)}."
        )
    if probabilities.shape != logits.shape:
        raise ValueError(f"Fold {fold} {subset} probabilities shape disagrees with logits.")
    if true_labels.shape != (expected_count,) or predicted_labels.shape != (expected_count,):
        raise ValueError(f"Fold {fold} {subset} label array dimensions are invalid.")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError(f"Fold {fold} {subset} raw outputs contain nonfinite values.")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float32), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"Fold {fold} {subset} probability rows do not sum to 1.")
    if not np.array_equal(predicted_labels, logits.argmax(axis=1)):
        raise ValueError(f"Fold {fold} {subset} predictions disagree with logits.")

    try:
        metadata = pd.read_csv(
            metadata_path,
            dtype={name: "string" for name in ("OBJECTID", "species", "image_path", "image_filename")},
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold} {subset} metadata cannot be read: {metadata_path}"
        ) from exc
    required_columns = {
        "row_index",
        "fold",
        "subset",
        "OBJECTID",
        "species",
        "encoded_true_label",
        "encoded_predicted_label",
        "image_path",
        "image_filename",
        "model",
        "learning_rate",
        "resize_strategy",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "evaluation_scope",
        "quarantined_final_holdout_used",
    }
    missing = sorted(required_columns.difference(metadata.columns))
    if missing or len(metadata) != expected_count:
        raise ValueError(
            f"Fold {fold} {subset} metadata count/columns are invalid; "
            f"rows={len(metadata)}, expected={expected_count}, missing={missing}."
        )
    if metadata["row_index"].astype(int).tolist() != list(range(expected_count)):
        raise ValueError(f"Fold {fold} {subset} metadata row indices are invalid.")
    if not (metadata["fold"].astype(int) == fold).all():
        raise ValueError(f"Fold {fold} {subset} metadata fold values are invalid.")
    if not (metadata["subset"].astype(str) == subset).all():
        raise ValueError(f"Fold {fold} {subset} metadata subset values are invalid.")
    if not (metadata["model"].astype(str) == args.model).all():
        raise ValueError(f"Fold {fold} {subset} metadata model values are invalid.")
    if not np.allclose(
        metadata["learning_rate"].astype(float), args.learning_rate, rtol=0.0, atol=1e-15
    ):
        raise ValueError(f"Fold {fold} {subset} metadata learning rates are invalid.")
    if not (metadata["resize_strategy"].astype(str) == args.resize_strategy).all():
        raise ValueError(f"Fold {fold} {subset} metadata resize strategies are invalid.")
    batch_expectations = {
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
    }
    for column, expected in batch_expectations.items():
        if not (metadata[column].astype(int) == expected).all():
            raise ValueError(
                f"Fold {fold} {subset} metadata {column} values are invalid."
            )
    expected_scope = (
        OUTER_TEST_DESCRIPTION
        if subset == "test"
        else "Inner validation subset used for checkpoint selection."
    )
    if not (metadata["evaluation_scope"].astype(str) == expected_scope).all():
        raise ValueError(f"Fold {fold} {subset} metadata evaluation scope is invalid.")
    holdout_values = metadata["quarantined_final_holdout_used"].astype(str).str.casefold()
    if not holdout_values.isin({"false"}).all():
        raise ValueError(f"Fold {fold} {subset} metadata incorrectly marks holdout use.")
    if not np.array_equal(metadata["encoded_true_label"].to_numpy(), true_labels):
        raise ValueError(f"Fold {fold} {subset} metadata true labels are misaligned.")
    if not np.array_equal(metadata["encoded_predicted_label"].to_numpy(), predicted_labels):
        raise ValueError(f"Fold {fold} {subset} metadata predictions are misaligned.")
    expected_labels = expected_df["Label"].to_numpy(dtype=np.int64)
    if not np.array_equal(true_labels, expected_labels):
        raise ValueError(f"Fold {fold} {subset} sample labels disagree with assignments.")
    expected_ids = expected_df["OBJECTID"].astype(str).tolist()
    if metadata["OBJECTID"].astype(str).tolist() != expected_ids:
        raise ValueError(f"Fold {fold} {subset} OBJECTID order disagrees with assignments.")
    expected_species = expected_df["Species"].astype(str).tolist()
    if metadata["species"].astype(str).tolist() != expected_species:
        raise ValueError(f"Fold {fold} {subset} species order disagrees with assignments.")
    expected_paths = expected_df["ImagePath"].astype(str).tolist()
    if metadata["image_path"].astype(str).tolist() != expected_paths:
        raise ValueError(f"Fold {fold} {subset} image paths disagree with assignments.")
    expected_filenames = expected_df["ImageFilename"].astype(str).tolist()
    if metadata["image_filename"].astype(str).tolist() != expected_filenames:
        raise ValueError(f"Fold {fold} {subset} filenames disagree with assignments.")
    return arrays


def validate_metric_artifact(
    path: Path,
    fold: int,
    subset: str,
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    metrics = load_json(path)
    expected = {
        "fold": fold,
        "subset": subset,
        "sample_count": int(arrays["true_labels"].shape[0]),
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
    }
    mismatches = [
        f"{name}: saved={metrics.get(name)!r}, expected={value!r}"
        for name, value in expected.items()
        if not configuration_value_matches(metrics.get(name), value)
    ]
    for name in ("loss", "accuracy", "macro_f1"):
        try:
            if not np.isfinite(float(metrics[name])):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            mismatches.append(f"{name}: missing or nonfinite")
    if mismatches:
        raise ValueError(
            f"Fold {fold} {subset} metric artifact mismatch:\n" + "\n".join(mismatches)
        )
    labels = arrays["true_labels"]
    predictions = arrays["predicted_labels"]
    saved_accuracy = float(metrics["accuracy"])
    saved_macro_f1 = float(metrics["macro_f1"])
    saved_loss = float(metrics["loss"])
    calculated_accuracy = float(accuracy_score(labels, predictions))
    calculated_macro_f1 = float(
        f1_score(labels, predictions, average="macro", zero_division=0)
    )
    calculated_loss = float(
        nn.functional.cross_entropy(
            torch.from_numpy(arrays["logits"]),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
        ).item()
    )
    if not np.isclose(saved_loss, calculated_loss, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Fold {fold} {subset} saved loss is inconsistent.")
    if not np.isclose(saved_accuracy, calculated_accuracy, rtol=0.0, atol=1e-12):
        raise ValueError(f"Fold {fold} {subset} saved accuracy is inconsistent.")
    if not np.isclose(saved_macro_f1, calculated_macro_f1, rtol=0.0, atol=1e-12):
        raise ValueError(f"Fold {fold} {subset} saved macro F1 is inconsistent.")
    return metrics


def validate_confusion_matrix_artifacts(
    csv_path: Path,
    png_path: Path,
    fold: int,
    subset: str,
    num_classes: int,
    expected_count: int,
) -> None:
    try:
        matrix = pd.read_csv(csv_path, index_col=0).to_numpy()
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold} {subset} confusion-matrix CSV cannot be read: {csv_path}"
        ) from exc
    try:
        matrix = matrix.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Fold {fold} {subset} confusion-matrix values are not numeric."
        ) from exc
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"Fold {fold} {subset} confusion-matrix shape is invalid.")
    if (
        not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.equal(matrix, np.floor(matrix)).all()
        or int(matrix.sum()) != expected_count
    ):
        raise ValueError(f"Fold {fold} {subset} confusion-matrix counts are invalid.")
    if not png_path.is_file() or png_path.stat().st_size <= 0:
        raise ValueError(f"Fold {fold} {subset} confusion-matrix PNG is missing or empty.")


def validate_completed_fold(
    directories: ArtifactDirectories,
    fold: int,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    spec: ModelSpec,
    crop_size: Optional[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    paths = fold_artifact_paths(directories, fold)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Fold {fold} has a completion record but is missing artifacts: {missing}"
        )
    fold_result = load_json(paths["fold_result"])
    result_expected = {
        "fold": fold,
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "crop_size_pixels": crop_size,
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation_steps": spec.grad_accum_steps,
        "effective_batch_size": spec.effective_batch_size,
        "mixed_precision_training": MIXED_PRECISION_TRAINING,
        "training_precision": TRAINING_PRECISION,
        "validation_sample_count": len(validation_df),
        "test_sample_count": len(test_df),
    }
    mismatches = [
        f"{name}: saved={fold_result.get(name)!r}, expected={value!r}"
        for name, value in result_expected.items()
        if not configuration_value_matches(fold_result.get(name), value)
    ]
    if mismatches:
        raise ValueError(
            f"Fold {fold} result-record mismatch:\n" + "\n".join(mismatches)
        )
    checkpoint = validate_checkpoint_artifact(
        paths["checkpoint"],
        fold,
        args,
        spec,
        crop_size,
        class_names,
        len(validation_df),
        len(test_df),
    )
    if int(checkpoint["best_epoch"]) != int(fold_result.get("best_epoch", -1)):
        raise ValueError(f"Fold {fold} checkpoint epoch disagrees with the fold result.")
    validate_epoch_metrics_artifact(
        paths["epoch_metrics"], fold, fold_result, args, spec, crop_size
    )
    validation_arrays = validate_raw_output_artifacts(
        paths["validation_outputs"],
        paths["validation_metadata"],
        validation_df,
        fold,
        "validation",
        args,
        len(class_names),
    )
    test_arrays = validate_raw_output_artifacts(
        paths["test_outputs"],
        paths["test_metadata"],
        test_df,
        fold,
        "test",
        args,
        len(class_names),
    )
    validation_metrics = validate_metric_artifact(
        paths["validation_metrics"],
        fold,
        "validation",
        args,
        validation_arrays,
    )
    test_metrics = validate_metric_artifact(
        paths["test_metrics"],
        fold,
        "test",
        args,
        test_arrays,
    )
    result_metric_pairs = {
        "best_validation_loss": validation_metrics["loss"],
        "best_validation_accuracy": validation_metrics["accuracy"],
        "best_validation_macro_f1": validation_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    for name, expected in result_metric_pairs.items():
        if not configuration_value_matches(fold_result.get(name), float(expected)):
            raise ValueError(f"Fold {fold} result metric {name} is inconsistent.")
    validate_confusion_matrix_artifacts(
        paths["validation_confusion_csv"],
        paths["validation_confusion_png"],
        fold,
        "validation",
        len(class_names),
        len(validation_df),
    )
    validate_confusion_matrix_artifacts(
        paths["test_confusion_csv"],
        paths["test_confusion_png"],
        fold,
        "test",
        len(class_names),
        len(test_df),
    )
    del checkpoint
    return fold_result


def metric_artifact_payload(
    metrics: Dict[str, Any],
    fold: int,
    subset: str,
    sample_count: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "fold": fold,
        "subset": subset,
        "sample_count": sample_count,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "outer_test_semantics": OUTER_TEST_DESCRIPTION if subset == "test" else None,
        "quarantined_final_holdout_used": False,
    }


def summarize_results(
    results_df: pd.DataFrame,
    args: argparse.Namespace,
    preprocessing: Dict[str, Any],
    assignment_summary: Dict[str, Any],
    directories: ArtifactDirectories,
) -> Dict[str, Any]:
    if results_df.empty:
        raise ValueError("Cannot generate summary from empty fold results.")

    def statistics(column: str) -> Dict[str, float]:
        values = results_df[column].to_numpy(dtype=float)
        return {
            "mean": float(np.mean(values)),
            "population_standard_deviation": float(np.std(values, ddof=0)),
        }

    return {
        "completed_at": utc_timestamp(),
        "results_directory": str(directories.results),
        "outputs_directory": str(directories.outputs),
        "checkpoints_directory": str(directories.checkpoints),
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "folds_completed": int(len(results_df)),
        "folds_expected": int(args.limit_folds or args.folds),
        "test_accuracy": statistics("test_accuracy"),
        "test_macro_f1": statistics("test_macro_f1"),
        "best_validation_accuracy": statistics("best_validation_accuracy"),
        "best_validation_macro_f1": statistics("best_validation_macro_f1"),
        "best_validation_loss": statistics("best_validation_loss"),
        "best_epoch": statistics("best_epoch"),
        "fixed_training_settings": {
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "checkpoint_selection_metric": "validation macro F1",
            "checkpoint_tolerance": CHECKPOINT_TOLERANCE,
            "loss": "torch.nn.CrossEntropyLoss()",
            "label_smoothing": 0,
            "optimizer": "AdamW",
            "weight_decay": 0,
            "scheduler": None,
            "mixed_precision_training": MIXED_PRECISION_TRAINING,
            "training_precision": TRAINING_PRECISION,
            "torch_compile": False,
            "data_augmentation": False,
            "class_weighting": False,
            "sampling_based_class_balancing": False,
            "focal_loss": False,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
        },
        "preprocessing": preprocessing,
        "split_assignment": assignment_summary,
        "outer_test_semantics": OUTER_TEST_DESCRIPTION,
        "every_eligible_tree_outer_tested_exactly_once": True,
        "quarantined_final_holdout_used": False,
        "retry_behavior": (
            "An interruption retry resumes the current batch attempt, validates every "
            "artifact for completed folds, and reruns only incomplete folds. A CUDA OOM "
            "discards all artifacts for this configuration attempt and restarts all folds "
            "at the next smaller micro-batch while retaining effective batch size 64."
        ),
    }


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize CUDA OOM exceptions, including errors wrapped with a cause/context."""

    seen: set[int] = set()
    current: Optional[BaseException] = error
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if cuda_oom_type and isinstance(current, cuda_oom_type):
            return True
        message = str(current).casefold()
        if "out of memory" in message and (
            "cuda" in message or "cublas" in message or "cudnn" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def clear_cuda_after_oom() -> None:
    """Release unreachable tensors and cached CUDA blocks between full-run attempts."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def remove_configuration_artifacts(args: argparse.Namespace) -> List[str]:
    """Remove only this configuration's three directory trees before an OOM retry."""

    directories = configuration_artifact_directories(args)
    roots_and_targets = (
    (args.experiment_results_dir.expanduser().resolve(), directories.results, "results"),
    (args.experiment_results_dir.expanduser().resolve(), directories.outputs, "outputs"),
    (args.checkpoints_dir.expanduser().resolve(), directories.checkpoints, ""),
    )

    removed: List[str] = []
    for root, target, subdirectory in roots_and_targets:
        expected = (
            root / args.model / args.experiment_id / subdirectory
        ).resolve()
        resolved_target = target.resolve()
        if resolved_target != expected or resolved_target == root:
            raise RuntimeError(
                "Refusing configuration cleanup because the target is not the exact "
                f"model/configuration child of its artifact root: {resolved_target}"
            )
        if target.is_symlink():
            raise RuntimeError(f"Refusing to recursively remove symlinked directory: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
        elif target.exists():
            raise RuntimeError(f"Expected configuration artifact path to be a directory: {target}")
    return removed


def existing_batch_fallback_state(
    args: argparse.Namespace,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Resume the batch size already recorded for an interrupted/completed attempt."""

    config_path = configuration_artifact_directories(args).results / "run_config.json"
    if not config_path.is_file():
        return 0, []
    config = load_json(config_path)
    try:
        pair = (
            int(config["micro_batch_size"]),
            int(config["gradient_accumulation_steps"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Existing configuration lacks a valid batch setting: {config_path}"
        ) from exc
    if pair not in BATCH_FALLBACK_SCHEDULE:
        raise ValueError(
            f"Existing configuration has unsupported batch setting {pair}: {config_path}"
        )
    fallback = config.get("batch_fallback", {})
    if not isinstance(fallback, dict):
        raise ValueError(f"Existing batch-fallback metadata is malformed: {config_path}")
    prior = fallback.get("prior_cuda_oom_attempts")
    if prior is None:
        attempts = fallback.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError(f"Existing batch-fallback history is malformed: {config_path}")
        prior = [
            item
            for item in attempts
            if isinstance(item, dict) and item.get("outcome") == "cuda_out_of_memory"
        ]
    if not isinstance(prior, list) or not all(isinstance(item, dict) for item in prior):
        raise ValueError(f"Existing batch-fallback history is malformed: {config_path}")
    return BATCH_FALLBACK_SCHEDULE.index(pair), list(prior)


def record_batch_fallback_success(
    args: argparse.Namespace,
    run_dir: Path,
    history: List[Dict[str, Any]],
    successful_pair: Tuple[int, int],
) -> None:
    success_record = {
        "attempt_number": BATCH_FALLBACK_SCHEDULE.index(successful_pair) + 1,
        "micro_batch_size": successful_pair[0],
        "gradient_accumulation_steps": successful_pair[1],
        "effective_batch_size": successful_pair[0] * successful_pair[1],
        "outcome": "success",
        "completed_at": utc_timestamp(),
    }
    complete_history = [*history, success_record]
    payload = {
        "enabled": True,
        "schedule": [
            {
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "effective_batch_size": micro_batch_size * grad_accum_steps,
            }
            for micro_batch_size, grad_accum_steps in BATCH_FALLBACK_SCHEDULE
        ],
        "attempts": complete_history,
        "successful_micro_batch_size": successful_pair[0],
        "successful_gradient_accumulation_steps": successful_pair[1],
        "effective_batch_size": successful_pair[0] * successful_pair[1],
    }
    config_path = run_dir / "run_config.json"
    config = load_json(config_path)
    config["batch_fallback"] = payload
    save_json(config_path, config)
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path)
    summary["batch_fallback"] = payload
    save_json(summary_path, summary)
    save_json(run_dir / "batch_fallback_history.json", payload)


def run_with_batch_fallback(args: argparse.Namespace, registry_spec: ModelSpec) -> Path:
    """Run one configuration, restarting it only when a CUDA OOM is encountered."""

    if not args.auto_batch_fallback:
        return run_training(args, registry_spec)

    start_index, oom_history = existing_batch_fallback_state(args)
    if start_index:
        micro_batch_size, grad_accum_steps = BATCH_FALLBACK_SCHEDULE[start_index]
        print(
            "Resuming the existing configuration at its recorded batch attempt: "
            f"micro={micro_batch_size}, accumulation={grad_accum_steps}.",
            flush=True,
        )

    for schedule_index in range(start_index, len(BATCH_FALLBACK_SCHEDULE)):
        micro_batch_size, grad_accum_steps = BATCH_FALLBACK_SCHEDULE[schedule_index]
        attempt_args = argparse.Namespace(**vars(args))
        attempt_args.batch_size = micro_batch_size
        attempt_args.grad_accum_steps = grad_accum_steps
        attempt_args.batch_fallback_attempt_number = schedule_index + 1
        attempt_args.batch_fallback_prior_oom_attempts = list(oom_history)
        spec = registry_spec.with_batch_configuration(
            micro_batch_size, grad_accum_steps
        )
        print(
            "Batch fallback attempt "
            f"{schedule_index + 1}/{len(BATCH_FALLBACK_SCHEDULE)}: "
            f"micro={micro_batch_size}, accumulation={grad_accum_steps}, "
            "effective=64, precision=float32.",
            flush=True,
        )

        oom_description: Optional[str] = None
        try:
            run_dir = run_training(attempt_args, spec)
        except Exception as error:
            if not is_cuda_out_of_memory(error):
                raise
            oom_description = f"{type(error).__name__}: {error}"[:2000]

        if oom_description is None:
            record_batch_fallback_success(
                attempt_args,
                run_dir,
                oom_history,
                (micro_batch_size, grad_accum_steps),
            )
            return run_dir

        oom_record = {
            "attempt_number": schedule_index + 1,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "effective_batch_size": micro_batch_size * grad_accum_steps,
            "outcome": "cuda_out_of_memory",
            "failed_at": utc_timestamp(),
            "error": oom_description,
        }
        oom_history.append(oom_record)
        clear_cuda_after_oom()
        removed = remove_configuration_artifacts(attempt_args)
        print(
            f"CUDA OOM at micro={micro_batch_size}, accumulation={grad_accum_steps}. "
            f"Removed {len(removed)} configuration-scoped artifact directories so the "
            "next attempt cannot mix results from different batch settings.",
            file=sys.stderr,
            flush=True,
        )
        for path in removed:
            print(f"  removed: {path}", file=sys.stderr, flush=True)

        if schedule_index + 1 == len(BATCH_FALLBACK_SCHEDULE):
            attempted = ", ".join(
                f"{micro}x{accumulation}"
                for micro, accumulation in BATCH_FALLBACK_SCHEDULE
            )
            raise RuntimeError(
                "Configuration failed with CUDA out of memory at every permitted batch "
                f"setting ({attempted})."
            )

    raise AssertionError("Unreachable batch-fallback state.")


def run_training(args: argparse.Namespace, spec: ModelSpec) -> Path:
    assignment_path = args.fold_assignments_path.expanduser().resolve()
    print(OUTER_TEST_DESCRIPTION, flush=True)
    print(
        "Every eligible subject is in exactly one outer test "
        "subset; the quarantined final holdout is not read by resize/LR experiment.",
        flush=True,
    )
    data_folder, csv_path, photos_path = resolve_data_paths(args)
    tree_df, encoder = load_tree_dataframe(csv_path)
    image_df = melt_and_validate_images(tree_df, photos_path)
    if args.use_preassigned_folds:
        tree_assignments, assignment_summary = (
            prepare_or_validate_assignments(
                assignment_path,
                tree_df,
                args.seed,
                args.folds,
                args.inner_folds,
                create_if_missing=False,
            )
        )
    else:
        tree_assignments = generate_tree_assignments(
            tree_df,
            args.seed,
            args.folds,
            args.inner_folds,
        )

        assignment_summary = validate_tree_assignments(
            tree_assignments,
            tree_df,
            args.seed,
            args.folds,
            args.inner_folds,
        )
        assignment_summary["path"] = None
        assignment_summary["identifier"] = ASSIGNMENT_VERSION
    image_assignments = make_image_assignments(image_df, tree_assignments, args.folds)

    crop_size = crop_size_for_strategy(args)
    preflight_crop_dimensions(image_df, crop_size, args)
    transform, preprocessing = build_transform(
        spec,
        args.resize_strategy,
        args.big_crop_size,
        args.small_crop_size,
    )
    probe = transform(Image.new("RGB", (1280, 960), color=(127, 127, 127)))
    if tuple(probe.shape[-2:]) != tuple(preprocessing["final_input_dimensions"]):
        raise ValueError(
            f"Transform output shape {tuple(probe.shape[-2:])} does not match "
            f"model input {preprocessing['final_input_dimensions']}."
        )

    if not torch.cuda.is_available():
        job_context = (
            f" SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}"
            if os.environ.get("SLURM_JOB_ID")
            else ""
        )
        raise RuntimeError(
            "CUDA is not available; resize/LR experiment training requires a GPU."
            + job_context
        )
    device = torch.device("cuda")
    seed_everything(args.seed)

    directories, timestamp_token, run_directory_created = make_run_directories(args)
    run_dir = directories.results
    checkpoint_dir = directories.checkpoints

    config: Dict[str, Any] = {
        "status": "running",
        "started_at": utc_timestamp(),
        "timestamp_token": timestamp_token,
        "results_directory": str(directories.results),
        "outputs_directory": str(directories.outputs),
        "checkpoints_directory": str(directories.checkpoints),
        "command": [sys.executable, *sys.argv],
        "command_line_args": command_line_args(args),
        "seed": args.seed,
        "data_folder": str(data_folder),
        "csv_path": str(csv_path),
        "photos_path": str(photos_path),
        "tree_count": int(len(tree_df)),
        "image_count": int(len(image_df)),
        "num_classes": int(len(encoder.classes_)),
        "class_name_order": encoder.classes_.tolist(),
        "experiment_id": args.experiment_id,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resize_strategy": args.resize_strategy,
        "crop_size_pixels": crop_size,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_selection_metric": "validation macro F1",
        "checkpoint_tie_breaking": (
            "higher validation macro F1; within 1e-12 lower validation loss; "
            "if both tie retain the earlier checkpoint"
        ),
        "outer_folds": args.folds,
        "inner_folds": args.inner_folds,
        "folds_requested": args.limit_folds or args.folds,
        "outer_test_semantics": OUTER_TEST_DESCRIPTION,
        "every_eligible_tree_outer_tested_exactly_once": True,
        "quarantined_final_holdout_used": False,
        "quarantined_final_holdout_policy": (
            "The separate quarantined final holdout dataset is excluded from resize/LR experiment."
        ),
        "split_implementation": (
            "tree-level StratifiedKFold(5) outer split plus first split from a "
            "tree-level StratifiedKFold(8) inner split"
        ),
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation_steps": spec.grad_accum_steps,
        "effective_batch_size": spec.effective_batch_size,
        "evaluation_batch_size": spec.eval_batch_size,
        "loss": "torch.nn.CrossEntropyLoss()",
        "label_smoothing": 0,
        "optimizer": "AdamW",
        "weight_decay": 0,
        "learning_rate_scheduler": None,
        "mixed_precision_training": MIXED_PRECISION_TRAINING,
        "training_precision": TRAINING_PRECISION,
        "torch_compile": False,
        "data_augmentation": False,
        "class_weighting": False,
        "sampling_based_class_balancing": False,
        "focal_loss": False,
        "preprocessing": preprocessing,
        "split_assignment": assignment_summary,
        "device": str(device),
        "torch_version": torch.__version__,
        "slurm": slurm_metadata(),
        "environment": {
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "batch_fallback": {
            "enabled": bool(getattr(args, "auto_batch_fallback", False)),
            "schedule": [
                {
                    "micro_batch_size": micro_batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "effective_batch_size": micro_batch_size * grad_accum_steps,
                }
                for micro_batch_size, grad_accum_steps in BATCH_FALLBACK_SCHEDULE
            ],
            "current_attempt_number": int(
                getattr(args, "batch_fallback_attempt_number", 1)
            ),
            "prior_cuda_oom_attempts": list(
                getattr(args, "batch_fallback_prior_oom_attempts", [])
            ),
        },
    }
    if run_directory_created:
        label_map = pd.DataFrame(
            {"encoded_label": range(len(encoder.classes_)), "species": encoder.classes_}
        )
        label_map.to_csv(run_dir / "label_map.csv", index=False, lineterminator="\n")
        save_image_assignments(
            image_assignments, run_dir / "image_level_split_assignments.csv"
        )
        save_json(
            run_dir / "tree_split_assignment_reference.json",
            {
                "shared_assignment_path": assignment_summary["path"],
                "shared_assignment_identifier": assignment_summary["identifier"],
                "tree_rows": len(tree_assignments),
                "outer_test_semantics": OUTER_TEST_DESCRIPTION,
                "every_eligible_tree_outer_tested_exactly_once": True,
                "quarantined_final_holdout_used": False,
                "note": (
                    "The assignment CSV contains one row per eligible development-dataset "
                    "tree and all five outer cross-validation subsets."
                ),
            },
        )
        save_json(run_dir / "run_config.json", config)
    else:
        existing_config = load_json(run_dir / "run_config.json")
        validate_run_config_compatibility(
            existing_config,
            args,
            spec,
            crop_size,
            len(encoder.classes_),
        )
        validate_saved_image_assignments(
            run_dir / "image_level_split_assignments.csv", image_assignments
        )
        config = existing_config
        print(
            f"Retry/resume detected for stable run directory: {run_dir}",
            flush=True,
        )

    print(f"Using device: {device}")
    print(f"Trees: {len(tree_df)}; images: {len(image_df)}; classes: {len(encoder.classes_)}")
    print(f"Shared split: {assignment_summary['identifier']}")
    print(f"Results directory: {directories.results}")
    print(f"Raw-output directory: {directories.outputs}")
    print(f"Checkpoint directory: {directories.checkpoints}")
    print(
        f"Training precision: {TRAINING_PRECISION}; "
        f"mixed precision enabled: {MIXED_PRECISION_TRAINING}"
    )
    print(
        f"Batch configuration: micro={spec.micro_batch_size}, "
        f"accumulation={spec.grad_accum_steps}, effective={spec.effective_batch_size}"
    )

    fold_results: List[Dict[str, Any]] = []
    fold_limit = args.limit_folds or args.folds
    for fold in range(1, fold_limit + 1):
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        train_df = fold_split(image_assignments, fold, "train")
        validation_df = fold_split(image_assignments, fold, "validation")
        test_df = fold_split(image_assignments, fold, "test")
        print(
            f"\nFold {fold}: train_images={len(train_df)}, "
            f"validation_images={len(validation_df)}, test_images={len(test_df)}"
        )
        paths = fold_artifact_paths(directories, fold)
        if paths["fold_result"].is_file():
            try:
                resumed_result = validate_completed_fold(
                    directories,
                    fold,
                    validation_df,
                    test_df,
                    args,
                    spec,
                    crop_size,
                    encoder.classes_.tolist(),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Fold {fold} has a completion record but failed resume validation. "
                    "No artifacts were deleted; inspect or remove only this fold's artifacts "
                    "before retrying."
                ) from exc
            fold_results.append(resumed_result)
            pd.DataFrame(fold_results).to_csv(
                run_dir / "fold_results.csv", index=False, lineterminator="\n"
            )
            print(
                f"Fold {fold} is complete and passed artifact validation; skipping training.",
                flush=True,
            )
            continue
        interrupted_paths = [
            path for path in paths.values() if path.exists()
        ]
        if interrupted_paths:
            removed = clean_interrupted_fold_artifacts(directories, fold)
            print(
                f"Fold {fold} has no completion record. Removed {len(removed)} "
                "fold-specific interrupted artifacts and will rerun only this fold:",
                flush=True,
            )
            for removed_path in removed:
                print(f"  removed: {removed_path}", flush=True)
        config["status"] = "running"
        config["last_resume_at"] = utc_timestamp()
        config["currently_training_fold"] = fold
        save_json(run_dir / "run_config.json", config)

        train_loader = get_dataloader(
            train_df,
            transform,
            spec.micro_batch_size,
            args.num_workers,
            True,
            device,
            fold_seed,
        )
        validation_loader = get_dataloader(
            validation_df,
            transform,
            spec.eval_batch_size,
            args.num_workers,
            False,
            device,
            fold_seed,
        )
        test_loader = get_dataloader(
            test_df,
            transform,
            spec.eval_batch_size,
            args.num_workers,
            False,
            device,
            fold_seed,
        )

        model, weight_cache_info = create_model(
            spec,
            len(encoder.classes_),
            device,
            require_cached_weights=args.require_cached_imagenet_weights,
        )
        if "weight_cache" not in config:
            config["weight_cache"] = weight_cache_info
            save_json(run_dir / "run_config.json", config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0
        )

        best_macro_f1 = float("-inf")
        best_loss = float("inf")
        best_epoch = 0
        best_selection_reason = ""
        epochs_without_f1_improvement = 0
        early_stopped = False
        epoch_rows: List[Dict[str, Any]] = []
        checkpoint_path = checkpoint_dir / f"fold_{fold}_best.pt"
        epoch_metrics_path = run_dir / f"fold_{fold}_epoch_metrics.csv"

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                spec.grad_accum_steps,
            )
            validation_metrics = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                len(encoder.classes_),
                capture_outputs=False,
                context=f"fold {fold} epoch {epoch} validation",
            )
            select_checkpoint, f1_improved, selection_reason = checkpoint_decision(
                validation_metrics["macro_f1"],
                validation_metrics["loss"],
                best_macro_f1,
                best_loss,
            )

            if select_checkpoint:
                best_macro_f1 = float(validation_metrics["macro_f1"])
                best_loss = float(validation_metrics["loss"])
                best_epoch = epoch
                best_selection_reason = selection_reason
                checkpoint_payload = {
                    "fold": fold,
                    "experiment_id": args.experiment_id,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "best_validation_loss": best_loss,
                    "checkpoint_selection_reason": best_selection_reason,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mixed_precision_training": MIXED_PRECISION_TRAINING,
                    "training_precision": TRAINING_PRECISION,
                    "model_name": args.model,
                    "imagenet_weight_enum": preprocessing["weight_enum"],
                    "resolved_imagenet_weight": preprocessing["resolved_weight"],
                    "learning_rate": args.learning_rate,
                    "resize_strategy": args.resize_strategy,
                    "crop_size_pixels": crop_size,
                    "micro_batch_size": spec.micro_batch_size,
                    "gradient_accumulation_steps": spec.grad_accum_steps,
                    "effective_batch_size": spec.effective_batch_size,
                    "class_name_order": encoder.classes_.tolist(),
                    "num_classes": len(encoder.classes_),
                    "expected_validation_samples": len(validation_df),
                    "expected_test_samples": len(test_df),
                    "command_line_args": command_line_args(args),
                    "split_assignment_identifier": assignment_summary["identifier"],
                    "preprocessing": preprocessing,
                }
                torch.save(checkpoint_payload, checkpoint_path)

            if f1_improved:
                epochs_without_f1_improvement = 0
            else:
                epochs_without_f1_improvement += 1

            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "learning_rate": args.learning_rate,
                "model": args.model,
                "resize_strategy": args.resize_strategy,
                "crop_size_pixels": crop_size,
                "micro_batch_size": spec.micro_batch_size,
                "gradient_accumulation_steps": spec.grad_accum_steps,
                "effective_batch_size": spec.effective_batch_size,
                "mixed_precision_training": MIXED_PRECISION_TRAINING,
                "training_precision": TRAINING_PRECISION,
                "checkpoint_selected": select_checkpoint,
                "checkpoint_selection_reason": selection_reason,
                "epochs_without_validation_f1_improvement": epochs_without_f1_improvement,
            }
            epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(
                epoch_metrics_path, index=False, lineterminator="\n"
            )
            print(
                f"Fold {fold} epoch {epoch}: train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['accuracy']:.6f} "
                f"train_macro_f1={train_metrics['macro_f1']:.6f} "
                f"val_loss={validation_metrics['loss']:.6f} "
                f"val_acc={validation_metrics['accuracy']:.6f} "
                f"val_macro_f1={validation_metrics['macro_f1']:.6f}; {selection_reason}"
            )

            if epochs_without_f1_improvement >= args.patience:
                early_stopped = True
                print(
                    f"Early stopping fold {fold} at epoch {epoch}: validation macro F1 "
                    f"did not improve for {args.patience} consecutive epochs."
                )
                break

        if not checkpoint_path.is_file() or best_epoch == 0:
            raise FileNotFoundError(f"Best checkpoint was not created for fold {fold}.")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        best_validation = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            len(encoder.classes_),
            capture_outputs=True,
            context=f"fold {fold} retained checkpoint validation",
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            len(encoder.classes_),
            capture_outputs=True,
            context=f"fold {fold} retained checkpoint test",
        )
        validation_npz, validation_metadata = save_raw_outputs(
            best_validation,
            validation_df,
            directories.outputs,
            fold,
            "validation",
            checkpoint,
            args,
            len(encoder.classes_),
        )
        test_npz, test_metadata = save_raw_outputs(
            test_metrics,
            test_df,
            directories.outputs,
            fold,
            "test",
            checkpoint,
            args,
            len(encoder.classes_),
        )
        validation_confusion_csv, validation_confusion_png = save_confusion_matrix(
            best_validation["true_labels"],
            best_validation["predicted_labels"],
            encoder.classes_.tolist(),
            run_dir / f"fold_{fold}_best_validation_confusion_matrix",
        )
        test_confusion_csv, test_confusion_png = save_confusion_matrix(
            test_metrics["true_labels"],
            test_metrics["predicted_labels"],
            encoder.classes_.tolist(),
            run_dir / f"fold_{fold}_test_confusion_matrix",
        )
        validation_metrics_path = paths["validation_metrics"]
        test_metrics_path = paths["test_metrics"]
        save_json(
            validation_metrics_path,
            metric_artifact_payload(
                best_validation,
                fold,
                "validation",
                len(validation_df),
                args,
            ),
        )
        save_json(
            test_metrics_path,
            metric_artifact_payload(
                test_metrics,
                fold,
                "test",
                len(test_df),
                args,
            ),
        )

        fold_result = {
            "fold": fold,
            "experiment_id": args.experiment_id,
            "best_epoch": int(checkpoint["best_epoch"]),
            "epochs_completed": len(epoch_rows),
            "early_stopped": early_stopped,
            "early_stopping_status": (
                "stopped_after_patience"
                if early_stopped
                else "maximum_epochs_completed"
            ),
            "checkpoint_selection_reason": checkpoint[
                "checkpoint_selection_reason"
            ],
            "selection_validation_macro_f1": checkpoint[
                "best_validation_macro_f1"
            ],
            "selection_validation_loss": checkpoint["best_validation_loss"],
            "best_validation_loss": best_validation["loss"],
            "best_validation_accuracy": best_validation["accuracy"],
            "best_validation_macro_f1": best_validation["macro_f1"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "learning_rate": args.learning_rate,
            "model": args.model,
            "resize_strategy": args.resize_strategy,
            "crop_size_pixels": crop_size,
            "micro_batch_size": spec.micro_batch_size,
            "gradient_accumulation_steps": spec.grad_accum_steps,
            "effective_batch_size": spec.effective_batch_size,
            "evaluation_batch_size": spec.eval_batch_size,
            "mixed_precision_training": MIXED_PRECISION_TRAINING,
            "training_precision": TRAINING_PRECISION,
            "imagenet_weight_enum": preprocessing["weight_enum"],
            "resolved_imagenet_weight": preprocessing["resolved_weight"],
            "input_height": preprocessing["final_input_dimensions"][0],
            "input_width": preprocessing["final_input_dimensions"][1],
            "normalization_mean": json.dumps(preprocessing["normalization_mean"]),
            "normalization_std": json.dumps(preprocessing["normalization_std"]),
            "interpolation_mode": preprocessing["interpolation_mode"],
            "antialias": json.dumps(preprocessing["antialias"]),
            "checkpoint_path": str(checkpoint_path),
            "raw_validation_outputs_path": str(validation_npz),
            "raw_validation_metadata_path": str(validation_metadata),
            "raw_test_outputs_path": str(test_npz),
            "raw_test_metadata_path": str(test_metadata),
            "validation_metrics_path": str(validation_metrics_path),
            "test_metrics_path": str(test_metrics_path),
            "validation_confusion_matrix_csv": str(validation_confusion_csv),
            "validation_confusion_matrix_png": str(validation_confusion_png),
            "test_confusion_matrix_csv": str(test_confusion_csv),
            "test_confusion_matrix_png": str(test_confusion_png),
            "split_assignment_identifier": assignment_summary["identifier"],
            "validation_sample_count": len(validation_df),
            "test_sample_count": len(test_df),
            "outer_test_semantics": OUTER_TEST_DESCRIPTION,
            "quarantined_final_holdout_used": False,
        }
        # This fold-specific record is the completion marker and is written only
        # after every required fold artifact has been created.
        save_json(paths["fold_result"], fold_result)
        validated_fold_result = validate_completed_fold(
            directories,
            fold,
            validation_df,
            test_df,
            args,
            spec,
            crop_size,
            encoder.classes_.tolist(),
        )
        fold_results.append(validated_fold_result)
        pd.DataFrame(fold_results).to_csv(
            run_dir / "fold_results.csv", index=False, lineterminator="\n"
        )
        print(
            f"Fold {fold} retained checkpoint: epoch={checkpoint['best_epoch']}, "
            f"validation_macro_f1={best_validation['macro_f1']:.6f}; "
            f"test_accuracy={test_metrics['accuracy']:.6f}, "
            f"test_macro_f1={test_metrics['macro_f1']:.6f}"
        )

        del model, optimizer, train_loader, validation_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(fold_results)
    if results_df.empty:
        raise ValueError("No folds completed; refusing to create an empty summary.")
    summary = summarize_results(
        results_df, args, preprocessing, assignment_summary, directories
    )
    save_json(run_dir / "summary.json", summary)
    config["status"] = "completed"
    config["completed_at"] = summary["completed_at"]
    config["folds_completed"] = int(len(results_df))
    config.pop("currently_training_fold", None)
    save_json(run_dir / "run_config.json", config)
    print("\nSummary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


def assignment_mode(args: argparse.Namespace, create_if_missing: bool) -> None:
    _data_folder, csv_path, photos_path = resolve_data_paths(args)
    tree_df, _encoder = load_tree_dataframe(csv_path)
    image_df = melt_and_validate_images(tree_df, photos_path)
    assignments, summary = prepare_or_validate_assignments(
        args.fold_assignments_path,
        tree_df,
        args.seed,
        args.folds,
        args.inner_folds,
        create_if_missing=create_if_missing,
    )
    if len(assignments) != tree_df["OBJECTID"].nunique():
        raise ValueError("Tree assignment file is not one row per eligible tree.")
    print(f"Validated {len(image_df)} referenced image paths.")
    print(json.dumps(summary, indent=2, sort_keys=True))


def resolve_run_seed(args: argparse.Namespace) -> int:
    """Choose once, reusing persisted seeds before any folds or models are built."""
    sources: List[Tuple[str, int]] = []

    def add_seed(source: str, value: Any) -> None:
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError(f"Invalid seed in {source}: {value!r}")
        number = int(value)
        if not 0 <= number <= 4_294_967_290:
            raise ValueError(f"Seed in {source} must be between 0 and 4294967290.")
        sources.append((source, number))

    if args.seed is not None:
        add_seed("--seed", args.seed)
    training_mode = not any((
        args.validate_data_only, args.prepare_fold_assignments_only,
        args.validate_fold_assignments_only, args.verify_transforms_only,
        args.verify_imagenet_weights_only,
    ))
    if training_mode:
        config_path = configuration_artifact_directories(args).results / "run_config.json"
        if config_path.is_file():
            config = load_json(config_path)
            saved_seed = config.get("seed", config.get("command_line_args", {}).get("seed"))
            add_seed(str(config_path), saved_seed)
    if any((args.use_preassigned_folds, args.prepare_fold_assignments_only,
            args.validate_fold_assignments_only)):
        path = args.fold_assignments_path.expanduser().resolve()
        if path.is_file():
            assignments = load_assignment_csv(path)
            if "seed" not in assignments or assignments["seed"].nunique(dropna=False) != 1:
                raise ValueError(f"Assignments must contain one consistent seed: {path}")
            add_seed(str(path), assignments["seed"].iloc[0])
    if len({value for _, value in sources}) > 1:
        raise ValueError(
            "Requested and saved seeds disagree. Use the saved seed when resuming, "
            "or a new experiment ID for a new run: " + repr(sources)
        )
    args.seed = sources[0][1] if sources else random.SystemRandom().randint(0, 4_294_967_290)
    print(f"Seed: {args.seed}" + (" (reused/requested)" if sources else " (new random seed)"), flush=True)
    return args.seed


def validate_saved_image_assignments(path: Path, image_assignments: pd.DataFrame) -> None:
    """Reject dataset or split changes even when no fold has completed yet."""
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = image_assignments.rename(columns={
        "Species": "species", "Label": "encoded_label", "ImagePath": "image_path",
        "ImageFilename": "image_filename", "Photo": "photo_column",
    })
    columns = ["fold", "subset", "OBJECTID", "species", "encoded_label",
               "image_path", "image_filename", "photo_column"]
    expected = expected.loc[:, columns].astype(str).reset_index(drop=True)
    if list(saved.columns) != columns or not saved.reset_index(drop=True).equals(expected):
        raise ValueError(
            "The dataset paths, labels, image list, or fold assignments changed since this run "
            "was created. Use a new experiment ID or a separate project folder. "
            "Existing results were not overwritten."
        )


def validate_data_mode(args: argparse.Namespace) -> None:
    """CPU-only checks suitable for a Slurm preparation job."""
    _data_folder, csv_path, photos_path = resolve_data_paths(args)
    subjects, encoder = load_tree_dataframe(csv_path)
    images = melt_and_validate_images(subjects, photos_path)
    if args.use_preassigned_folds:
        assignments, summary = prepare_or_validate_assignments(
            args.fold_assignments_path, subjects, args.seed, args.folds,
            args.inner_folds, create_if_missing=False,
        )
    else:
        assignments = generate_tree_assignments(subjects, args.seed, args.folds, args.inner_folds)
        summary = validate_tree_assignments(assignments, subjects, args.seed, args.folds, args.inner_folds)
    preflight_crop_dimensions(images, crop_size_for_strategy(args), args)
    for path in images["ImagePath"]:
        try:
            with Image.open(path) as source:
                source.load()
                converted = ImageOps.exif_transpose(source).convert("RGB")
                converted.load()
        except Exception as exc:
            raise ValueError(f"Cannot decode image as RGB: {path}") from exc
    print(json.dumps({
        "validated_subjects": len(subjects), "validated_images": len(images),
        "class_names": encoder.classes_.tolist(), "seed": args.seed,
        "crop_size": crop_size_for_strategy(args), "fold_validation": summary,
    }, indent=2))


def main() -> None:
    args = parse_args()
    spec = validate_args(args)
    if args.verify_transforms_only:
        results = verify_all_transforms()
        print(json.dumps({"verified_transform_count": len(results), "results": results}, indent=2))
        return
    if args.verify_imagenet_weights_only:
        verify_all_imagenet_weights()
        return
    resolve_run_seed(args)
    if args.validate_data_only:
        validate_data_mode(args)
        return
    if args.prepare_fold_assignments_only:
        assignment_mode(args, create_if_missing=True)
        return
    if args.validate_fold_assignments_only:
        assignment_mode(args, create_if_missing=False)
        return
    run_with_batch_fallback(args, spec)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
