#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/test/output}"
ZIP_NAME="${2:-output.zip}"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "Missing output dir: $OUT_DIR" >&2
  exit 1
fi

rm -f "$ZIP_NAME"
mkdir -p /tmp/submission/output
cp "$OUT_DIR"/*.json /tmp/submission/output/
(
  cd /tmp/submission
  zip -r "$OLDPWD/$ZIP_NAME" output
)
echo "Created $ZIP_NAME"
