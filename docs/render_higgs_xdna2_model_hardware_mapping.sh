#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$script_dir/higgs_xdna2_model_hardware_mapping.tex"
out="$script_dir/higgs_xdna2_model_hardware_mapping.pdf"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required to render $src" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cp "$src" "$tmpdir/"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$tmpdir" \
  "$tmpdir/$(basename "$src")" >/dev/null
cp "$tmpdir/$(basename "${src%.tex}.pdf")" "$out"

echo "Wrote $out"
