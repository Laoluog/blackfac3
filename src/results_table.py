"""Aggregate per-model RFW results into markdown + LaTeX comparison tables.

Inputs:
  --rfw-results: one or more JSON files produced by evaluate_rfw.py
  --commercial: optional JSON produced by evaluate_commercial.py
Outputs:
  results_table.md   markdown table
  results_table.tex  LaTeX table

The table reports TAR@FAR=0.01 per race plus the gap between African and
Caucasian for each model. Best value per column is bolded. Literature
baselines (ArcFace, AdaFace) are included from a hardcoded dict — see
LITERATURE_BASELINES below and verify before publishing.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RACES = ("African", "Asian", "Caucasian", "Indian")
METRIC_KEY = "TAR@FAR=0.01"

# Hardcoded literature baselines (TAR@FAR=1e-2 on RFW).
# NOTE: numbers are placeholders gathered from common citations; verify
# against the actual papers before quoting in publication.
LITERATURE_BASELINES: dict[str, dict[str, float]] = {
    "ArcFace (paper)": {
        "African": 0.8398,
        "Asian": 0.9218,
        "Caucasian": 0.9415,
        "Indian": 0.9028,
    },
    "AdaFace (paper)": {
        "African": 0.9410,
        "Asian": 0.9620,
        "Caucasian": 0.9740,
        "Indian": 0.9510,
    },
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_rfw_result(path: Path) -> tuple[str, dict[str, float]]:
    """Load an evaluate_rfw.py JSON and return (model_name, {race: TAR@FAR=0.01})."""
    with open(path) as f:
        data = json.load(f)
    name = data.get("model", path.stem)
    per_race = {}
    for race in RACES:
        m = data["per_race"].get(race, {})
        per_race[race] = float(m.get(METRIC_KEY, float("nan")))
    return name, per_race


def load_commercial(path: Path) -> dict[str, dict[str, float]]:
    """Load evaluate_commercial.py JSON. Returns {provider: {race: TAR}} (single combined value)."""
    with open(path) as f:
        data = json.load(f)
    races = data.get("races", [])
    out: dict[str, dict[str, float]] = {}
    for prov, m in data.get("providers", {}).items():
        v = float(m.get(METRIC_KEY, float("nan")))
        # Commercial run is typically over a single race subset; spread the
        # combined value across each race actually evaluated.
        out[f"{prov} (commercial)"] = {r: v if r in races else float("nan") for r in RACES}
    return out


def add_gap(per_race: dict[str, float]) -> float:
    """Return Caucasian - African gap (positive = bias against African)."""
    try:
        return per_race["Caucasian"] - per_race["African"]
    except KeyError:
        return float("nan")


def fmt(v: float, bold: bool = False) -> str:
    """Format a float to 4 decimals (or '--' if NaN); optionally bold for markdown."""
    if v != v:  # NaN
        return "--"
    s = f"{v:.4f}"
    return f"**{s}**" if bold else s


def fmt_tex(v: float, bold: bool = False) -> str:
    """LaTeX float formatter mirroring `fmt`."""
    if v != v:
        return "--"
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_per_col(rows: list[tuple[str, dict[str, float], float]]) -> dict[str, float]:
    """Compute the best (max) value per race column and the *min* absolute gap."""
    best: dict[str, float] = {}
    for race in RACES:
        vals = [pr[race] for _, pr, _ in rows if pr[race] == pr[race]]
        best[race] = max(vals) if vals else float("nan")
    gaps = [abs(g) for _, _, g in rows if g == g]
    best["gap"] = min(gaps) if gaps else float("nan")
    return best


def build_rows(
    rfw_paths: list[Path], commercial_path: Path | None
) -> list[tuple[str, dict[str, float], float]]:
    """Combine literature baselines + measured runs into [(name, {race: TAR}, gap)]."""
    rows: list[tuple[str, dict[str, float], float]] = []
    for name, per_race in LITERATURE_BASELINES.items():
        rows.append((name, per_race, add_gap(per_race)))
    for p in rfw_paths:
        name, per_race = load_rfw_result(p)
        rows.append((name, per_race, add_gap(per_race)))
    if commercial_path is not None:
        for name, per_race in load_commercial(commercial_path).items():
            rows.append((name, per_race, add_gap(per_race)))
    return rows


def to_markdown(rows: list[tuple[str, dict[str, float], float]]) -> str:
    """Render rows as a GitHub-flavored markdown table with bold best cells."""
    best = best_per_col(rows)
    head = ["Model", *RACES, "Caucasian-African gap"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, pr, gap in rows:
        cells = [name]
        for race in RACES:
            cells.append(fmt(pr[race], bold=(pr[race] == best[race])))
        cells.append(fmt(gap, bold=(abs(gap) == best["gap"]) if gap == gap else False))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(rows: list[tuple[str, dict[str, float], float]]) -> str:
    """Render rows as a LaTeX tabular environment with bold best cells."""
    best = best_per_col(rows)
    col_spec = "l" + "c" * (len(RACES) + 1)
    out = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Model & " + " & ".join(RACES) + " & Caucasian-African gap \\\\",
        "\\midrule",
    ]
    for name, pr, gap in rows:
        cells = [name]
        for race in RACES:
            cells.append(fmt_tex(pr[race], bold=(pr[race] == best[race])))
        cells.append(fmt_tex(gap, bold=(abs(gap) == best["gap"]) if gap == gap else False))
        out.append(" & ".join(cells) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rfw-results",
        nargs="+",
        required=True,
        type=Path,
        help="One or more *_rfw_results.json files from evaluate_rfw.py",
    )
    p.add_argument(
        "--commercial",
        type=Path,
        default=None,
        help="Optional commercial_results.json from evaluate_commercial.py",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.rfw_results, args.commercial)
    md = to_markdown(rows)
    tex = to_latex(rows)
    md_path = args.output_dir / "results_table.md"
    tex_path = args.output_dir / "results_table.tex"
    md_path.write_text(md)
    tex_path.write_text(tex)
    logger.info("Wrote %s and %s", md_path, tex_path)
    print(md)


if __name__ == "__main__":
    main()
