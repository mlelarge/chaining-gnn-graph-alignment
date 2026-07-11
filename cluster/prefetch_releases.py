#!/usr/bin/env python3
"""Pre-download all pretrained GitHub releases into <checkpoint-dir>/<tag>/.

Run this on a machine WITH internet (e.g. the CLEPS login node). The compute
nodes then reuse the cached checkpoints offline — `reproduce_results` /
`download_release` skip the download when `config.json` + a `.ckpt` are present.

Uses only the standard library (no torch), so it is safe to run on a login node.

    python cluster/prefetch_releases.py [checkpoint_dir]   # default ./checkpoints
"""

import json
import os
import sys
import urllib.request

REPO = "mlelarge/chaining-gnn-graph-alignment"
TAGS = [
    # synthetic (tab:ER-Reg) + the transferred ChFGNN-ER4
    "v1.0.0-er500-d4-pn0.22",
    "v1.0.0-er500-d80-pn0.24",
    "v1.0.0-reg500-d10-pn0.11",
    # real-world dataset-specific ChFGNN
    "v1.1.0-canetscience-pn0.1",
    "v1.1.0-canetscience-pn0.2",
    "v1.1.0-euroroad-pn0.1",
    "v1.1.0-euroroad-pn0.2",
    "v1.1.0-yeast25lc-pn0.05",
    "v1.1.0-yeast25lc-pn0.1",
    "v1.1.0-multimagna",
]


def fetch(tag, ckpt_dir):
    dest = os.path.join(ckpt_dir, tag)
    os.makedirs(dest, exist_ok=True)
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={"Accept": "application/vnd.github.v3+json"})
    release = json.loads(urllib.request.urlopen(req).read().decode())
    for asset in release["assets"]:
        out = os.path.join(dest, asset["name"])
        if os.path.exists(out):
            print(f"    have {asset['name']}")
            continue
        print(f"    download {asset['name']}")
        urllib.request.urlretrieve(asset["browser_download_url"], out)


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints"
    for tag in TAGS:
        print(f"[{tag}]")
        fetch(tag, ckpt_dir)
    print(f"\nAll releases cached under {ckpt_dir}")


if __name__ == "__main__":
    main()
