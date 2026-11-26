"""T-SEC-1 (NFR-04): no key patterns in tracked files; the Alpaca key env vars are never set in files.

CI additionally runs detect-secrets against .secrets.baseline (see .github/workflows/ci.yml).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from atu2.core.credentials import CREDENTIAL_ENV_VARS
from atu2.core.logging import KEY_PATTERNS

REPO = Path(__file__).resolve().parents[2]
SCANNED_ENV_VARS = (*CREDENTIAL_ENV_VARS, "NASDAQ_DATA_LINK_API_KEY")
# A literal value (quoted or not, 8+ chars) assigned to one of the key variables, e.g. in a
# shell export, .env line, YAML or Python. References like $VAR, <placeholder> or getpass() are fine.
ENV_ASSIGNMENT = re.compile(r"\b(?:" + "|".join(SCANNED_ENV_VARS) + r")\b\s*[:=]\s*['\"]?(?![$<{(])[^'\"\s#,)]{8,}")
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
    assert scan_text(f"key = {fake_id}")
    assert not scan_text("key = os.environ['ALPACA_API_KEY']")


@pytest.mark.parametrize("name", ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"])
def test_scanner_covers_all_four_credential_names(name):
    value = "Ab1-" + "x9" * 8
    for line in (f"{name} = '{value}'", f'export {name}="{value}"', f"{name}={value}", f"{name}: {value}"):
        assert scan_text(line), line
    for safe in (f"export {name}=$SOMETHING", f"{name}=<your key>", f'os.environ["{name}"] = getpass()', f'"{name}",'):
        assert not scan_text(safe), safe


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
