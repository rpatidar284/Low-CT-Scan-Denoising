#!/usr/bin/env bash
set -euo pipefail

# Offline pseudo-label generation on NDCT volumes only.
# Inputs expected:
#   data/ndct_volumes_nifti/<case_id>.nii.gz
# Outputs:
#   data/totalseg_raw/<case_id>.nii.gz
#
# Requires TotalSegmentator installed in your environment.

INPUT_DIR="${1:-data/ndct_volumes_nifti}"
OUTPUT_DIR="${2:-data/totalseg_raw}"
TASK="${3:-total}"

mkdir -p "${OUTPUT_DIR}"

for nii in "${INPUT_DIR}"/*.nii.gz; do
  [ -f "${nii}" ] || continue
  case_id="$(basename "${nii}" .nii.gz)"
  out_case_dir="${OUTPUT_DIR}/${case_id}"
  mkdir -p "${out_case_dir}"

  # total segmentator writes multiple files in a folder; we'll keep one merged file too.
  TotalSegmentator -i "${nii}" -o "${out_case_dir}" --task "${TASK}"

  # If TotalSegmentator generated a combined multilabel map, copy it to canonical location.
  # Common output filename is "segmentations.nii.gz" in some setups.
  if [ -f "${out_case_dir}/segmentations.nii.gz" ]; then
    cp "${out_case_dir}/segmentations.nii.gz" "${OUTPUT_DIR}/${case_id}.nii.gz"
  fi
done

echo "TotalSegmentator offline run complete."

