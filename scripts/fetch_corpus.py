"""Download the spec chapters and figures. Run: python -m scripts.fetch_corpus"""

import argparse
import time
from pathlib import Path

import httpx

from app.corpus import BASE_URL, PAGES, RAW_DIR, RAW_FIGURES_DIR, figure_names

PAUSE_SECONDS = 2.0
TIMEOUT_SECONDS = 30.0
USER_AGENT = "riscv-rag-assistant/0.1 (educational project; one-off spec download)"  # ASCII only


class Downloader:
    """Download missing files, pausing between real requests."""

    def __init__(self, client: httpx.Client, force: bool):
        self.client = client
        self.force = force
        self.requested = False

    def fetch(self, url: str, target: Path) -> None:
        if target.exists() and not self.force:
            print(f"{target.name}: already exists, skipping")
            return
        if self.requested:
            time.sleep(PAUSE_SECONDS)
        response = self.client.get(url)
        self.requested = True
        response.raise_for_status()
        target.write_bytes(response.content)
        print(f"{target.name}: {len(response.content)} bytes")


def fetch_all(force: bool = False) -> None:
    RAW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        downloader = Downloader(client, force)
        figures: dict[str, None] = {}
        for page in PAGES:
            target = RAW_DIR / f"{page}.html"
            downloader.fetch(f"{BASE_URL}/{page}.html", target)
            figures.update(dict.fromkeys(figure_names(target.read_text(encoding="utf-8"))))
        for name in figures:
            downloader.fetch(f"{BASE_URL}/_images/{name}", RAW_FIGURES_DIR / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download existing files again")
    fetch_all(parser.parse_args().force)
