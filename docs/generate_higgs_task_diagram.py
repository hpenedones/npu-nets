#!/usr/bin/env python3
"""Generate a standalone HIGGS physics-primer diagram for the appendix."""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path


WIDTH = 1320
HEIGHT = 680


def parse_args() -> argparse.Namespace:
    docs_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create a simple HIGGS signal/background explainer diagram for appendix use."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=docs_dir / "higgs_task_diagram.svg",
        help="Path for the generated SVG asset.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=docs_dir / "higgs_task_diagram.png",
        help="Optional PNG asset rendered from the SVG when ImageMagick is available.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=docs_dir / "higgs_task_diagram.pdf",
        help="Optional PDF asset rendered from the SVG when ImageMagick is available.",
    )
    return parser.parse_args()


def svg_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = "#0f172a",
    line_height: int = 32,
    anchor: str = "start",
) -> str:
    spans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_height
        spans.append(
            f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{"".join(spans)}</text>'
    )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: list[str],
    fill: str,
    stroke: str = "#1f2937",
    title_size: int = 24,
    body_size: int = 18,
    body_line_height: int = 22,
) -> str:
    title_lines = title.split("\n")
    body_y = y + 72 + 22 * (len(title_lines) - 1)
    parts = [
        (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
        ),
        svg_text(
            x + 24,
            y + 38,
            title_lines,
            size=title_size,
            weight=700,
            line_height=24,
        ),
        svg_text(
            x + 24,
            body_y,
            body,
            size=body_size,
            line_height=body_line_height,
        ),
    ]
    return "\n".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        'stroke="#334155" stroke-width="4" marker-end="url(#arrowhead)"/>'
    )


def elbow(start: tuple[float, float], mid_x: float, end: tuple[float, float]) -> str:
    x1, y1 = start
    x2, y2 = end
    return (
        f'<path d="M {x1} {y1} L {mid_x} {y1} L {mid_x} {y2} L {x2} {y2}" '
        'fill="none" stroke="#334155" stroke-width="4" marker-end="url(#arrowhead)"/>'
    )


def build_svg() -> str:
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Inter, Arial, sans-serif">'
        ),
        "<defs>",
        (
            '<marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" '
            'orient="auto" markerUnits="strokeWidth">'
            '<path d="M 0 0 L 12 6 L 0 12 z" fill="#334155"/></marker>'
        ),
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(
            WIDTH / 2,
            52,
            ["HIGGS benchmark: from reconstructed event to signal/background label"],
            size=34,
            weight=700,
            anchor="middle",
        ),
        svg_text(
            WIDTH / 2,
            88,
            ["Each row is one reconstructed collision event summarized as 28 kinematic features."],
            size=20,
            fill="#475569",
            anchor="middle",
        ),
        box(
            40,
            140,
            250,
            190,
            title="Reconstructed\n event",
            body=[
                "1 charged lepton",
                "missing pT",
                "up to 4 jets",
                "per-jet b-tag scores",
            ],
            fill="#e0f2fe",
            stroke="#0369a1",
            body_size=19,
            body_line_height=24,
        ),
        box(
            330,
            112,
            410,
            250,
            title="28 kinematic features",
            body=[
                "21 low-level observables",
                "7 derived masses",
                "lepton / jets: pT, η, φ",
                "missing pT magnitude + φ",
                "m_jj, m_lv, m_bb, ...",
            ],
            fill="#dbeafe",
            stroke="#1d4ed8",
            body_size=19,
            body_line_height=24,
        ),
        box(
            795,
            170,
            210,
            160,
            title="Classifier",
            body=[
                "outputs a score",
                "more signal-like",
                "or background-like",
                "repo model: residual MLP",
            ],
            fill="#ede9fe",
            stroke="#6d28d9",
            body_size=19,
            body_line_height=24,
        ),
        box(
            1050,
            86,
            230,
            150,
            title="Signal\n(label 1)",
            body=[
                "rare target process",
                "used to define",
                "positive examples",
            ],
            fill="#dcfce7",
            stroke="#15803d",
            body_size=19,
            body_line_height=24,
        ),
        box(
            1050,
            286,
            230,
            150,
            title="Background\n(label 0)",
            body=[
                "ordinary processes",
                "with a similar",
                "visible final state",
            ],
            fill="#fee2e2",
            stroke="#b91c1c",
            body_size=19,
            body_line_height=24,
        ),
        arrow(290, 235, 330, 235),
        arrow(740, 250, 795, 250),
        elbow((1005, 224), 1028, (1050, 160)),
        elbow((1005, 276), 1028, (1050, 360)),
        '<line x1="80" y1="472" x2="1240" y2="472" stroke="#cbd5e1" stroke-width="2"/>',
        svg_text(
            WIDTH / 2,
            525,
            [
                "pT = transverse momentum   •   η, φ = direction coordinates   •   b-tag = jet likely from a b quark",
            ],
            size=18,
            fill="#334155",
            line_height=24,
            anchor="middle",
        ),
        svg_text(
            WIDTH / 2,
            560,
            [
                "missing pT = momentum imbalance from invisible particles   •   m_jj, m_lv, m_bb = invariant-mass combinations",
            ],
            size=18,
            fill="#334155",
            line_height=24,
            anchor="middle",
        ),
        svg_text(
            WIDTH / 2,
            620,
            [
                "Physics goal: separate rare signal-like events from common look-alike background events.",
            ],
            size=22,
            weight=600,
            fill="#0f172a",
            anchor="middle",
        ),
        "</svg>",
    ]
    return "\n".join(parts)


def render_png(svg_path: Path, png_path: Path) -> bool:
    if shutil.which("convert") is None:
        return False
    try:
        subprocess.run(
            [
                "convert",
                "-density",
                "180",
                "-background",
                "white",
                str(svg_path),
                "-resize",
                f"{WIDTH}x{HEIGHT}",
                str(png_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def render_pdf(svg_path: Path, pdf_path: Path) -> bool:
    if shutil.which("convert") is None:
        return False
    try:
        subprocess.run(
            ["convert", str(svg_path), str(pdf_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def main() -> None:
    args = parse_args()
    svg_path = args.output.resolve()
    svg_path.write_text(build_svg(), encoding="utf-8")

    png_path = args.png_output.resolve()
    png_rendered = render_png(svg_path, png_path)
    pdf_path = args.pdf_output.resolve()
    pdf_rendered = render_pdf(svg_path, pdf_path)

    outputs = [str(svg_path)]
    if png_rendered:
        outputs.append(str(png_path))
    if pdf_rendered:
        outputs.append(str(pdf_path))
    print("Wrote " + ", ".join(outputs))


if __name__ == "__main__":
    main()
