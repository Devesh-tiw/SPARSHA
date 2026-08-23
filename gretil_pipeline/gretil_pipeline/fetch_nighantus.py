#!/usr/bin/env python3
"""Download every medicinal Nighantu currently available as plain text in GRETIL.

GRETIL currently exposes Rājanighaṇṭu and Aṣṭāṅganighaṇṭu. The script does not
silently substitute unrelated texts and explicitly excludes Bījanighaṇṭu.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.request import Request, urlopen

OUTPUT_DIR = Path(__file__).resolve().parent / "raw_texts"
USER_AGENT = "SPARSHA research corpus fetcher/1.0 (polite; two files only)"

SOURCES = {
    "rajanighantu.txt": (
        "https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/"
        "plaintext/sa_narahari-rAjanighaNTu.txt"
    ),
    "ashtanga_nighantu.txt": (
        "https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/"
        "plaintext/sa_vAhaTa-aSTAGganighaNTu.txt"
    ),
}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        data = response.read()
    text = data.decode("utf-8")
    if "# Text" not in text or "## Publisher:" not in text:
        raise RuntimeError(f"Downloaded file failed GRETIL header validation: {url}")
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, (filename, url) in enumerate(SOURCES.items()):
        if index:
            time.sleep(2)  # polite delay
        data = download(url)
        destination = OUTPUT_DIR / filename
        destination.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        print(f"Downloaded {filename}: {len(data):,} bytes, sha256={digest}")

    print("\nExcluded: Bījanighaṇṭu (Tantric mantra/bīja lexicon, not medicinal pharmacology).")
    print("Not present in the GRETIL corpus: Dhanvantari, Madanapāla, Kaiyadeva, Bhāvaprakāśa Nighaṇṭus.")
    print("Use the official NIIMH/CCRAS e-Nighantu pages for those texts; verify reuse terms before extraction.")


if __name__ == "__main__":
    main()
