#!/bin/bash
# Create the inputs the harness needs in work/ (safe to re-run):
#   corpus.txt  ~4 MB of Go standard-library source, for needle.py (enough for
#               ~1.2M tokens of prompt) and as a perplexity corpus for ppl.sh
#   test.png    a small synthetic image for needle.py --image
#   nccl-stub/libnccl.so.2  empty NCCL stub the ai-dock build needs to start (ppl.sh)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${CTXTEST_WORK:-$here/work}"
mkdir -p "$work/nccl-stub"
goroot="$(go env GOROOT 2>/dev/null || echo /usr/lib/golang)"
if [[ ! -s "$work/corpus.txt" ]]; then
    find "$goroot/src/net/http" "$goroot/src/encoding" "$goroot/src/strings" "$goroot/src/sort" \
         "$goroot/src/bufio" "$goroot/src/fmt" "$goroot/src/regexp" "$goroot/src/go" \
         -name '*.go' ! -name '*_test.go' | sort | xargs cat > "$work/corpus.txt"
fi
if [[ ! -s "$work/test.png" ]]; then
    python3 - "$work/test.png" <<'PY'
import sys
from PIL import Image, ImageDraw
im = Image.new("RGB", (800, 600), "white"); d = ImageDraw.Draw(im)
d.rectangle([100, 100, 400, 300], fill="red"); d.ellipse([450, 250, 700, 500], fill="blue")
d.text((120, 400), "SALE DEED 2026", fill="black")
im.save(sys.argv[1])
PY
fi
if [[ ! -s "$work/nccl-stub/libnccl.so.2" ]]; then
    echo 'void stub(void){}' | gcc -shared -fPIC -x c - -o "$work/nccl-stub/libnccl.so.2" -Wl,-soname,libnccl.so.2
fi
ls -la "$work/corpus.txt" "$work/test.png" "$work/nccl-stub/libnccl.so.2"
