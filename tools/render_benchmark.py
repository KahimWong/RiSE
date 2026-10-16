"""Render the paper benchmark as a GitHub-compatible SVG with colored digits."""

import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    margin, row_height, header_height = 24, 40, 86
    widths = [180, 100] + [110] * 10
    width = sum(widths) + 2 * margin
    height = header_height + row_height * len(data["rows"]) + 2 * margin
    xs = [margin]
    for value in widths:
        xs.append(xs[-1] + value)
    low, high = (0, 86, 190), (205, 54, 0)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(data["caption"])}</title>',
        '<desc id="description">F1 scores for fifteen methods across seven test sets. Digits use the paper’s blue-to-orange scale, with alternating gray rows. Bold marks the best and underline marks the second best. A copyable table follows in the README.</desc>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" font-size="19" fill="#18212b">',
    ]

    def text(
        x, y, label, anchor="middle", color="#18212b", bold=False, underline=False
    ):
        decoration = ' text-decoration="underline"' if underline else ""
        weight = ' font-weight="700"' if bold else ""
        svg.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}"{weight}{decoration}>{escape(label)}</text>'
        )

    def rule(y, thickness=1):
        svg.append(
            f'<line x1="{margin}" y1="{y}" x2="{xs[-1]}" y2="{y}" stroke="#78818b" stroke-width="{thickness}"/>'
        )

    rule(margin, 2)
    text((xs[2] + xs[6]) / 2, margin + 27, "In-Domain (Synthetic Data)", bold=True)
    text((xs[6] + xs[11]) / 2, margin + 27, "Cross-Domain (Real Data)", bold=True)
    headers = [
        "Method",
        "Venue",
        "FCD",
        "SCD",
        "Test",
        "AvgI",
        "T-SROIE",
        "OSTF",
        "TPIC-13",
        "RTM",
        "AvgC",
        "Avg",
    ]
    for i, label in enumerate(headers):
        text(
            xs[i] + 12 if i < 2 else (xs[i] + xs[i + 1]) / 2,
            margin + 66,
            label,
            anchor="start" if i < 2 else "middle",
            bold=True,
        )
    rule(margin + header_height, 1.5)
    for index, row in enumerate(data["rows"]):
        y = margin + header_height + index * row_height
        if index % 2 == 0:
            svg.append(
                f'<rect x="{margin}" y="{y}" width="{sum(widths)}" height="{row_height}" fill="#f4f5f7"/>'
            )
        if index in (6, 11):
            rule(y, 1.5)
        baseline = y + 26
        text(
            xs[0] + 12,
            baseline,
            row["method"],
            anchor="start",
            bold=row["venue"] == "Ours",
        )
        text(xs[1] + 12, baseline, row["venue"], anchor="start")
        for column, score in enumerate(row["scores"], 2):
            t = score["color_percent"] / 100
            rgb = tuple(round(a * (1 - t) + b * t) for a, b in zip(low, high))
            color = "#%02x%02x%02x" % rgb
            text(
                (xs[column] + xs[column + 1]) / 2,
                baseline,
                score["value"],
                color=color,
                bold=score["style"] == "bold",
                underline=score["style"] == "underline",
            )
    rule(height - margin, 2)
    svg.append("</g></svg>")
    return "\n".join(svg) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets/table2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/table2.svg")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(args.data.read_text())))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
