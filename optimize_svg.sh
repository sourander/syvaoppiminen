#!/usr/bin/env bash
#
# Usage: ./optimize_svg.sh docs/images/some_image_from_udlbook.svg

set -e

# --- Detect OS ---
if [[ "$OSTYPE" == "darwin"* ]]; then
  SED_INPLACE=(-i '')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  SED_INPLACE=(-i)
else
  echo "Unsupported OS: $OSTYPE"
  exit 1
fi

# --- Check argument ---
if [ $# -ne 1 ]; then
  echo "Usage: $0 <file.svg>"
  exit 1
fi

SVG_FILE="$1"

if [ ! -f "$SVG_FILE" ]; then
  echo "Error: File '$SVG_FILE' not found."
  exit 1
fi

# --- Check for svgo ---
if ! command -v svgo >/dev/null 2>&1; then
  echo "⚠️  svgo not found. Install it with:"
  echo "    brew install svgo"
  exit 1
fi

# --- Add background rect ---
# Insert the <rect> line immediately after the opening <svg ...> tag
sed "${SED_INPLACE[@]}" '/<svg/,/>/{
/>/s/>/>\
<rect width="100%" height="100%" fill="rgb(245,245,245)"\/>/
}' "$SVG_FILE"
echo "✅ Added background rect to $SVG_FILE"


# --- Optimize with svgo ---
svgo "$SVG_FILE" -o "$SVG_FILE"
echo "✅ Optimized SVG with svgo."
