#!/usr/bin/env bash
# Generate eight batch scripts from a manifest and your edited batch-1 script.
# Uses only Bash and Python 3; no Slurm jobs or training are started.
#
# Usage:
#   bash generate_run_scripts.sh MANIFEST.tsv RUN_BATCH_1.sh [BATCH_COUNT] [OUTPUT_DIR]
#
# Examples:
#   bash generate_run_scripts.sh resize_lr_experiment_manifest.tsv run_resize_lr_batch_1.sh
#
# Defaults: eight batches; OUTPUT_DIR is generated_runs beside RUN_BATCH_1.sh.
# The source template and manifest are never modified. Batch 1 keeps the
# template's configuration_indices; remaining rows are assigned to batches 2-N.
# Edit batch 1 and rerun this generator whenever you change shared job settings.

set -euo pipefail

usage() {
  echo "Usage: bash ${0##*/} MANIFEST.tsv RUN_BATCH_1.sh [BATCH_COUNT=8] [OUTPUT_DIR]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 2 || $# > 4 )); then
  usage >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python 3 is required. Run the generator where python3 is available." >&2
  exit 1
fi

manifest_path="$1"
template_path="$2"
batch_count="${3:-8}"
output_directory="${4:-}"

python3 - "${manifest_path}" "${template_path}" "${batch_count}" "${output_directory}" <<'PY'
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_write(path, content, executable=False):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        if executable:
            temporary_path.chmod(0o755)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    manifest = Path(sys.argv[1]).expanduser().resolve()
    template = Path(sys.argv[2]).expanduser().resolve()
    count = int(sys.argv[3])
    require(count >= 1, "BATCH_COUNT must be a positive integer.")
    require(manifest.is_file(), "Manifest file does not exist: " + str(manifest))
    require(template.is_file(), "Template script does not exist: " + str(template))
    output = (
        Path(sys.argv[4]).expanduser().resolve()
        if sys.argv[4] else template.parent / "generated_runs"
    )
    require(output != template.parent,
            "Use a separate output directory, such as generated_runs, to protect batch 1.")

    manifest_bytes = manifest.read_bytes()
    template_bytes = template.read_bytes()
    source = template_bytes.decode("utf-8").replace("\r\n", "\n")
    lines = manifest_bytes.decode("utf-8").splitlines()
    require(len(lines) >= 2, "The manifest must contain a header and at least one experiment.")
    columns = lines[0].split("\t")
    require(len(set(columns)) == len(columns), "The manifest contains duplicate column names.")
    require({"experiment_id", "model"}.issubset(columns),
            "The manifest must have experiment_id and model columns.")

    # The supplied batch scripts use an unquoted, single-line TSV header.
    header_pattern = re.compile(
        r"^[ \t]*if \[\[ \"\$\{header\}\" != \$'([^'\n]*)' \]\]; then[ \t]*$", re.MULTILINE
    )
    header_matches = list(header_pattern.finditer(source))
    require(len(header_matches) == 1, "Expected one manifest-header check in the template.")
    expected_columns = header_matches[0].group(1).replace(r"\t", "\t").split("\t")
    require(columns == expected_columns,
            "Manifest header does not match the template. Use the matching manifest and batch template.")

    rows = []
    seen_ids = set()
    for line_number, line in enumerate(lines[1:], start=2):
        require(bool(line.strip()), "Blank manifest line at line " + str(line_number))
        values = line.split("\t")
        require(len(values) == len(columns),
                "Incorrect number of TSV fields at line " + str(line_number))
        row = dict(zip(columns, values))
        for key in ("experiment_id", "model"):
            require(bool(row[key]) and row[key] == row[key].strip(),
                    "Empty or padded " + key + " at line " + str(line_number))
        require(row["experiment_id"] not in seen_ids,
                "Duplicate experiment ID: " + row["experiment_id"])
        seen_ids.add(row["experiment_id"])
        rows.append(row)

    filename_match = re.fullmatch(r"(.+_batch_)1\.sh", template.name)
    require(filename_match is not None, "Template filename must end with _batch_1.sh.")
    filename_prefix = filename_match.group(1)
    job_pattern = re.compile(r"^([ \t]*#SBATCH[ \t]+--job-name=)([^ \t\r\n]+)[ \t]*$", re.MULTILINE)
    job_matches = list(job_pattern.finditer(source))
    require(len(job_matches) == 1, "Expected one #SBATCH --job-name= directive.")
    job_name = job_matches[0].group(2)
    require(job_name.endswith("_1"), "The template's Slurm job name must end with _1.")
    job_prefix = job_name[:-1]
    indices_pattern = re.compile(
        r"^([ \t]*)configuration_indices=\(([^)]*)\)([ \t]*(?:#[^\n]*)?)$", re.MULTILINE
    )
    index_matches = list(indices_pattern.finditer(source))
    require(len(index_matches) == 1, "Expected one configuration_indices=(...) assignment.")
    tokens = index_matches[0].group(2).split()
    require(tokens and all(re.fullmatch(r"0|[1-9][0-9]*", token) for token in tokens),
            "Batch 1 must contain explicit, nonnegative integer configuration indices.")
    first_indices = [int(token) for token in tokens]
    require(len(first_indices) == len(set(first_indices)), "Batch 1 contains duplicate indices.")
    require(all(index < len(rows) for index in first_indices),
            "Batch 1 has an index outside this manifest; update its configuration_indices first.")
    first_set = set(first_indices)
    remaining = [index for index in range(len(rows)) if index not in first_set]
    require((count > 1 or not remaining) and len(remaining) >= count - 1,
            "The remaining manifest rows cannot fill the requested number of batches.")

    row_count_pattern = re.compile(
        r'^([ \t]*if \[\[ "\$\{manifest_rows\}" != ")[0-9]+(" \]\]; then[ \t]*)$', re.MULTILINE
    )
    require(len(list(row_count_pattern.finditer(source))) == 1,
            "Expected one numeric manifest_rows check in the template.")

    # Preserve batch 1; spread each model across the other jobs, breaking ties
    # by total configuration count. This balances counts, not estimated runtime.
    batches = [first_indices.copy()] + [[] for _ in range(count - 1)]
    model_counts = [Counter(rows[index]["model"] for index in indices) for indices in batches]
    models = list(dict.fromkeys(row["model"] for row in rows))
    for model in models:
        for index in (index for index in remaining if rows[index]["model"] == model):
            destination = min(
                range(1, count),
                key=lambda batch: (model_counts[batch][model], len(batches[batch]), batch),
            )
            batches[destination].append(index)
            model_counts[destination][model] += 1
    for indices in batches[1:]:
        indices.sort()
    assigned = [index for indices in batches for index in indices]
    require(sorted(assigned) == list(range(len(rows))),
            "Internal error: experiments were duplicated or omitted.")
    require(all(batches), "An empty batch was produced; reduce BATCH_COUNT.")

    generated = []
    for batch_number, indices in enumerate(batches, start=1):
        path = output / (filename_prefix + str(batch_number) + ".sh")
        text = source.replace(template.name, path.name)
        text = job_pattern.sub(lambda match: match.group(1) + job_prefix + str(batch_number), text)
        text = indices_pattern.sub(
            lambda match: match.group(1) + "configuration_indices=("
            + " ".join(str(index) for index in indices) + ")" + match.group(3), text,
        )
        text = re.sub(r"\b([Bb]atch)[ \t]+1\b",
                      lambda match: match.group(1) + " " + str(batch_number), text)
        text = re.sub(r"\bjob[ \t]+1[ \t]+of[ \t]+[0-9]+\b",
                      "job " + str(batch_number) + " of " + str(count), text)
        text = re.sub(r"\b([Bb]atch)[ \t]+" + str(batch_number) + r"[ \t]+of[ \t]+[0-9]+\b",
                      lambda match: match.group(1) + " " + str(batch_number) + " of " + str(count), text)
        text = row_count_pattern.sub(
            lambda match: match.group(1) + str(len(rows)) + match.group(2), text,
        )
        text = re.sub(r"(Manifest must contain exactly )[0-9]+( data rows;)",
                      lambda match: match.group(1) + str(len(rows)) + match.group(2), text)
        text = re.sub(r"^([ \t]*#[ \t]+)sbatch[ \t]+[^\n]*$",
                      lambda match: match.group(1) + "sbatch " + shlex.quote(str(path)),
                      text, flags=re.MULTILINE)
        check = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
        require(check.returncode == 0, "Generated script failed Bash syntax validation: "
                + path.name + "\n" + check.stderr)
        generated.append((path, text))

    timestamp = datetime.now(timezone.utc)
    summary = {
        "generated_at_utc": timestamp.isoformat(),
        "manifest": str(manifest),
        "template": str(template),
        "experiment_count": len(rows),
        "batch_count": count,
        "first_batch_indices_preserved": True,
        "batches": [
            {"script": str(path), "configuration_indices": indices,
             "experiment_ids": [rows[index]["experiment_id"] for index in indices],
             "model_counts": dict(Counter(rows[index]["model"] for index in indices))}
            for (path, _text), indices in zip(generated, batches)
        ],
    }
    writes = generated + [(output / "generation_summary.json", json.dumps(summary, indent=2) + "\n")]
    for path, _text in writes:
        require(not path.is_symlink(), "Refusing to replace a symlink: " + str(path))
        require(not path.exists() or path.is_file(), "Output path is not a regular file: " + str(path))

    output.mkdir(parents=True, exist_ok=True)
    backup_dir = output / "backups" / timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
    # Back up all changed files before replacing any of them.
    for path, text in writes:
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(backup_dir / path.name))
    for path, text in writes:
        if not path.exists() or path.read_bytes() != text.encode("utf-8"):
            atomic_write(path, text, executable=path.suffix == ".sh")
        elif path.suffix == ".sh":
            path.chmod(path.stat().st_mode | 0o111)

    for (path, _text), indices in zip(generated, batches):
        print(path.name + ": " + str(len(indices)) + " experiments")
    print("Verified: all " + str(len(rows)) + " experiments are assigned exactly once.")
    print("Generated scripts: " + str(output))
    print("Assignment details: " + str(output / "generation_summary.json"))
    if backup_dir.exists():
        print("Previous generated files backed up to: " + str(backup_dir))
    print("The source template and manifest were preserved. No jobs were submitted.")


try:
    main()
except (OSError, ValueError, UnicodeError) as error:
    print("Error: " + str(error), file=sys.stderr)
    sys.exit(1)
PY
