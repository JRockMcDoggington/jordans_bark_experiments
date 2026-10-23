# Resize and learning-rate experiment

Train and compare 11 pretrained EfficientNet models, three learning rates, and
three image-resizing strategies: **99 configurations, each with five folds**.
The models are EfficientNet B0–B7 and EfficientNetV2 S/M/L. Eight Slurm jobs share
the configurations; each configuration starts its own Python process.

## Quick start on Sapelo2

1. Upload/extract this entire folder into a writable project directory on Sapelo2.
2. Put your images in `image_dir/images/` and populate `image_dir/labels.csv`
   using the format below. The supplied CSV contains only the required header.
3. From a Sapelo2 login node:

   ```bash
   cd /path/to/resize_lr_experiment
   bash start.sh --run
   ```

This generates eight scripts under `generated_runs/`, queues one CPU preparation
job, and queues eight GPU jobs that wait for preparation to succeed. Preparation
checks the labels, every image, the 960-pixel crop requirement, and fold
feasibility. It downloads/verifies missing pretrained weights. If all expected
weight files are already nonempty/readable, it checks the dataset and leaves
weight loading to training. Use `--prepare --run` to verify all cached weights too.

Submission returns before training finishes. Check `sq --me` or
`squeue -u "$USER"`; logs are under `logs/`. A preparation failure prevents the
dependent training jobs from starting. Read `logs/prepare_<job-id>.err`, fix the
problem, and cancel the dependent jobs with `scancel <job-id> ...` before retrying.

The default GPU request is one **H100**, six CPU cores, 64 GB host RAM, and 48
hours **per batch job**. Eight jobs can therefore request eight GPUs in total.
Queue permissions, quotas, available hardware, and the size of your dataset
determine whether jobs start and finish within that time.

| `start.sh` option | Effect |
|---|---|
| No arguments, or `--prepare` | Queue CPU weight preparation only; does **not** start training. Can run before adding the dataset. |
| `--run` | Validate data, prepare missing weights, then run the full experiment. |
| `--prepare --run` | Validate data and verify/load all eleven cached weights before training. |
| `--reset` | Regenerate batch scripts and replace this project's pretrained-weight cache. Preserves datasets, results, and trained checkpoints. |
| `--reset --run` | Reset preparation, then run the experiment. |
| `--help` | Show startup options. |

`--run` and `--reset` refuse to proceed while this project has queued/running
jobs, preventing overlapping writes. Different experiments can use separate
project directories. The project defaults to the folder containing `start.sh`,
even when invoked from elsewhere; `EXPERIMENT_DIR` can override it.
`EXPERIMENT_RUN_SCRIPTS_DIR` optionally selects a different generated-script folder.

## Using your own images and labels

**The images do not have to depict trees.** Dog breeds, plant types, or other
image classes work. The output classifier is sized from the distinct class names
in your CSV; there is no fixed 35-class requirement or fixed total image count.

Keep these exact column names, even for a non-tree dataset:

```csv
OBJECTID,Species,Photo0,Photo1,Photo2,Photo3
dog_001,Beagle,beagle_001_front.jpg,beagle_001_side.jpg,,
dog_002,Beagle,beagle_002.jpg,,,
dog_101,Poodle,poodle_101.jpg,,,
```

These three example rows illustrate the format; they are not a sufficient dataset.

- `OBJECTID`: unique, nonempty **subject/group ID**, with exactly one row per
  subject. For dogs, use one ID per individual dog. All images of that dog must
  stay in its row so that the same dog never appears in both training and testing
  within a fold. For independent images with no shared subject, each image may
  have its own ID.
- `Species`: a nonempty class name, such as `Beagle`. Keep the header `Species`;
  the values can describe any classes. At least **two classes** are required.
- `Photo0`–`Photo3`: all four columns must exist. Each row needs **1–4 images**;
  leave unused cells empty. Extra photo columns are not used. Supporting more
  than four images per subject requires a code change; do not split one subject
  into several IDs to work around it.
- Each class needs **at least 10 independent subjects**, not merely 10 photos.
  Every fold checks for at least 7 training, 1 validation, and 2 test subjects per
  class. Class sizes and image counts per subject do not need to be equal.
- Image paths are relative to `image_dir/images/`, for example `dogs/a.jpg`.
  Use forward slashes. Absolute paths, paths outside the image directory, and
  repeated references to the same image file are rejected. Unreferenced files
  are ignored. Images copied under different filenames cannot be identified as
  duplicates automatically.
- Use images Pillow can decode, such as JPEG or PNG. Images are oriented using
  EXIF metadata and converted to RGB. For the **full supplied experiment**, every
  image must be at least **960 × 960 pixels**. Large crops use 960 × 960, small
  crops use 480 × 480, and padding is forbidden. A direct `resize`-only training
  command does not have the crop-size minimum.

The splits operate on subjects, with approximately 70% training, 10% validation,
and 20% test per fold. Reported accuracy and macro F1 are **image-level** metrics.
The five outer test sets are used for comparing configurations. Keep any separate
final holdout out of this dataset; directory components beginning with
`quarantined` are rejected as an additional guard.

## Training options

`start.sh` accepts only its startup options. To change options for all batch jobs,
edit the Python command in `run_resize_lr_batch_1.sh`, then regenerate:

```bash
bash generate_run_scripts.sh resize_lr_experiment_manifest.tsv run_resize_lr_batch_1.sh
```

Do this before submitting jobs. Existing generated scripts are reused by
`start.sh` when all eight exist. Edit the template rather than generated copies.
The template's `#SBATCH` lines control GPU type, memory, CPU count, and time.
No user-specific email address or account is included.

For a direct Python invocation, first obtain a suitable Sapelo2 compute
allocation, enter the project directory, and `source ./load_modules.sh`.
Do not train on a login node. `python training_script.py --help` lists all options.

| Training argument | Default and supported changes |
|---|---|
| `--model` | `efficientnet_b0`; B0–B7 and `efficientnet_v2_s`, `_m`, `_l` are supported. The manifest selects this for batch jobs. |
| `--learning-rate` | `0.001`; allowed values are `0.001`, `0.0001`, `0.0003`. The manifest selects this. |
| `--resize-strategy` | `resize`; also `big_center_crop_resize` or `small_center_crop_resize`. The manifest selects this. Final input dimensions follow each model's pretrained-weight preset. |
| `--epochs` | `80`; any positive maximum epoch count. |
| `--patience` | `10`; positive number of epochs without a macro-F1 improvement before early stopping. |
| `--num-workers` | `8` in Python; the batch template explicitly uses `4`. Any nonnegative integer; `0` loads images in the main process. |
| `--seed` | Random for a new configuration, saved and reused on resume; accepts `0` through `4294967290` inclusive. See below. |
| `--experiment-id` | `custom_run`; a unique output name without path separators. Batch jobs use the manifest IDs. Use a new ID/output directory when changing a completed run's settings. |
| `--data-folder` | `image_dir`; infers `labels.csv` and `images/` inside it. |
| `--csv-path`, `--photos-path` | Optional overrides for the CSV and image directory. Direct relative paths resolve from the current working directory. The launcher/template currently use `image_dir/labels.csv` and `image_dir/images`; keep both shell files in agreement if changing those locations. |
| `--experiment-results-dir` | `experiment_results`, relative to the working directory. |
| `--checkpoints-dir` | `checkpoints`, relative to the working directory; must not overlap the results directory. |
| `--batch-size`, `--grad-accum-steps` | `64` and `1`. Only these pairs are accepted: `64/1`, `32/2`, `16/4`, `8/8`. The effective batch target remains 64. |
| `--auto-batch-fallback` | Off in Python, enabled in the template. Starts at `64/1` and retries the entire configuration after a CUDA out-of-memory error using the next pair. Removes that configuration's previous attempt artifacts. |
| `--limit-folds` | Unset; all five folds. Set `1`–`5` for a shorter direct run. A partial run is excluded from the default five-fold report selection. |
| `--use-preassigned-folds` | Off. When enabled, require the saved assignment CSV and use its seed unless `--seed` explicitly agrees. |
| `--fold-assignments-path` | `fold_assignments.csv`; used only in assignment modes or with `--use-preassigned-folds`. |
| `--require-cached-imagenet-weights` | Off in Python, enabled in the template. Refuse network downloads during training; use the prepared cache. |

Although exposed as arguments, `--folds`, `--inner-folds`, `--big-crop-size`, and
`--small-crop-size` are **fixed by validation** at 5, 8, 960, and 480 respectively.
Other values require code changes. Optimizer, weight decay, augmentation, class
weighting, label smoothing, and mixed precision are not configurable CLI options.
This experiment uses AdamW, zero weight decay, ordinary cross-entropy, float32,
and no random image augmentation or class balancing.

CPU utility modes are mutually exclusive and exit without training:

| Mode | Purpose |
|---|---|
| `--validate-data-only` | Decode images and validate labels, splits, and the selected crop. Add `--resize-strategy big_center_crop_resize` to check full-grid image dimensions. |
| `--prepare-fold-assignments-only` | Create the assignment CSV if missing, otherwise validate it. |
| `--validate-fold-assignments-only` | Validate an existing assignment CSV. |
| `--verify-transforms-only` | Check all 33 model/resize combinations without images or pretrained downloads. |
| `--verify-imagenet-weights-only` | Download/load all eleven pretrained checkpoints. Uses `TORCH_HOME` when set. |

## Seeds, comparisons, and resuming

The default batch template chooses a fresh random seed **for each new
configuration**, so different configurations can use different folds. It records
that seed in `results/run_config.json`. Restarting an existing configuration
reuses its seed; OOM retries also retain it. Explicit conflicting seeds are rejected.
Seeding does not guarantee bit-for-bit results across hardware/software versions.

For comparisons using the same folds across every configuration, add `--seed 42`
to the template before the first run and regenerate the scripts. Alternatively,
in a CPU allocation with the modules loaded:

```bash
python training_script.py --prepare-fold-assignments-only --seed 42
```

Then add `--use-preassigned-folds` to the template and regenerate. The saved file's
seed is adopted automatically. Reusing subject splits makes configuration
comparisons easier to interpret.

After jobs finish or are cancelled, rerun `bash start.sh --run` to retry. Completed
folds are checked and skipped; an interrupted fold starts again from epoch one.
Resume checks reject changed settings, labels, image lists/paths, and splits.
They do not compare image bytes, so do not replace image contents in place while
resuming. `--reset` resets preparation, **not trained results**. For a genuinely
new experiment or dataset, use a fresh project/output directory.

## Outputs and report

| Location | Contents |
|---|---|
| `experiment_results/<model>/<experiment_id>/results/` | Configuration, seed, label map, split assignments, epoch/fold metrics, and confusion matrices. |
| `experiment_results/<model>/<experiment_id>/outputs/` | Validation/test logits, probabilities, labels, and matching image metadata. |
| `checkpoints/<model>/<experiment_id>/` | Best checkpoint for each fold, selected using validation macro F1 with validation loss as the tie-breaker. |
| `torch_cache/` | Downloaded ImageNet weights shared by this project's jobs. |
| `logs/` | Slurm standard output/error logs. |

After training, generate the Excel report with Python 3 (standard library only):

```bash
python3 generate_experiment_report.py
```

It reads `experiment_results/` beside the report script and writes
`resize_lr_fold_level_test_results.xlsx` beside that folder. Optional arguments:
`--results <folder-or-zip>`, `--output <file.xlsx>`, and `--expected-folds 5`.
Incomplete configurations are shown but excluded from model selection.
`reference_results/previous_random_split_results.xlsx` is the supplied historical
report, preserved byte-for-byte with its original labels. It is not output from
this package audit and is not an input to training/report generation.

## Software and local checks

`load_modules.sh` loads the shared Sapelo2 stack: Python 3.11.3, CUDA 12.1.1,
SciPy-bundle 2023.07, matplotlib 3.7.2, PyTorch 2.1.2, scikit-learn 1.4.2, and
torchvision 0.16.2. The bundle supplies NumPy/pandas; Pillow is also required.
If a module is retired, update the helper and verify a compatible stack on the
cluster. Training requires a CUDA GPU. CPU validation and tests do not.
Public cluster references: [GPU/software](https://wiki.gacrc.uga.edu/wiki/GPU)
and [job submission](https://wiki.gacrc.uga.edu/wiki/Running_Jobs_on_Sapelo2).

With training dependencies available, run `python -m unittest discover -s tests -v`
to repeat the local regression checks. They use temporary files, mock Slurm, and
a tiny CPU model for the full-loop test; no real jobs or pretrained downloads are
performed. See `AUDIT.md` for the changes and the limits of local verification.
