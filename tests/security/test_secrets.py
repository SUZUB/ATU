"""T-SEC-1 (NFR-04): no key patterns in tracked files; the Alpaca key env vars are never set in files.

CI additionally runs detect-secrets against .secrets.baseline (see .github/workflows/ci.yml).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from atu2.core.logging import KEY_PATTERNS

REPO = Path(__file__).resolve().parents[2]
# literal values assigned to key variables  # pragma: allowlist secret
ENV_ASSIGNMENT = re.compile(
    r"(?:ALPACA_API_(?:KEY|SECRET)|APCA_API_(?:KEY_ID|SECRET_KEY)|NASDAQ_DATA_LINK_API_KEY)\s*[:=]\s*['\"][A-Za-z0-9/+]{12,}['\"]"
)
SKIP_SUFFIXES = {".zip", ".png", ".jpg", ".jpeg", ".pdf", ".docx", ".pt", ".pth", ".parquet", ".pkl", ".joblib"}


def _tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    out = subprocess.run([git, "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    return [REPO / p for p in out.stdout.decode("utf-8").split("\0") if p]


def scan_text(text: str) -> list[str]:
    hits = [m.group(0)[:12] + "..." for pat in KEY_PATTERNS for m in pat.finditer(text)]
    hits += [m.group(0)[:24] + "..." for m in ENV_ASSIGNMENT.finditer(text)]
    return hits


def test_scanner_detects_planted_keys():
    fake_id = "PK" + "Z9" * 9
    fake_secret = "ALPACA_API_SECRET = '" + "Ab1" * 10 + "'"
    assert scan_text(f"key = {fake_id}")
    assert scan_text(fake_secret)
    assert not scan_text("key = os.environ['ALPACA_API_KEY']")


def test_no_key_patterns_in_tracked_files():
    findings = []
    for path in _tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings += [f"{path.relative_to(REPO)}: {hit}" for hit in scan_text(text)]
    assert findings == []
