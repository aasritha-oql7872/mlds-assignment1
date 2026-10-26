#!/usr/bin/env bash
# Build report/report.pdf from report/report.html with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html"
echo "wrote $PWD/report.pdf"
