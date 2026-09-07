import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("artifacts/results")


def save_result(experiment_name: str, payload: dict[str, Any]) -> Path:
    """Save one experiment result in a shared JSON schema."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{experiment_name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_results() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"))
