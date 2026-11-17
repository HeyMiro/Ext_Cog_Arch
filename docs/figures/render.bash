#!/usr/bin/env bash
# Regenerate the paper figures: SVG (source of truth), PDF (vector, for
# LaTeX) and PNG (2x, for the README). Needs python3 and Chrome/Chromium.
set -euo pipefail
cd "$(dirname "$0")"
python3 make_figures.py

CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)"
if [ -z "$CHROME" ]; then
	echo "Chrome/Chromium not found: SVGs written, PDF/PNG skipped" >&2
	exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for svg in fig*.svg; do
	name="${svg%.svg}"
	read -r w h < <(sed -n 's/^<svg[^>]* width="\([0-9]*\)" height="\([0-9]*\)".*/\1 \2/p' "$svg")
	cat > "$TMP/$name.html" <<EOF
<!doctype html><html><head><style>
@page { size: ${w}px ${h}px; margin: 0 }
html, body { margin: 0; padding: 0 }
svg { display: block }
</style></head><body>$(cat "$svg")</body></html>
EOF
	"$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
		--no-pdf-header-footer --print-to-pdf="$PWD/$name.pdf" "file://$TMP/$name.html" 2>/dev/null
	"$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars --force-device-scale-factor=2 \
		--window-size="$w,$h" --screenshot="$PWD/$name.png" "file://$TMP/$name.html" 2>/dev/null
	echo "rendered $name.pdf $name.png"
done
