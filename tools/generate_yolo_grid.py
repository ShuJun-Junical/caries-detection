#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample images and render YOLO labels into a 4x4 (or custom) grid."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/train/images"),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("dataset/train/labels"),
        help="Directory containing YOLO txt labels.",
    )
    parser.add_argument(
        "--rows", type=int, default=4, help="Number of grid rows."
    )
    parser.add_argument(
        "--cols", type=int, default=4, help="Number of grid columns."
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=320,
        help="Width/height of each tile in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (omit for true random each run).",
    )
    args = parser.parse_args()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"runs/dataset_sample_{args.rows}x{args.cols}.jpg"),
        help="Output image path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(f"runs/dataset_sample_{args.rows}x{args.cols}_manifest.txt"),
        help="Output manifest path containing selected image files.",
    )
    return parser.parse_args()


def get_font(size: int = 16) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_yolo_boxes(img: Image.Image, label_file: Path, font: ImageFont.ImageFont) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lines = label_file.read_text(encoding="utf-8").splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, bw, bh = parts[:5]
        if int(float(cls)) != 0:
            continue
        x, y, bw, bh = map(float, (x, y, bw, bh))

        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h

        color = (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"class {cls}"
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        by1 = max(0, y1 - th - 4)
        draw.rectangle([x1, by1, x1 + tw + 6, by1 + th + 4], fill=color)
        draw.text((x1 + 3, by1 + 2), label, fill=(0, 0, 0), font=font)

    return img


def has_class0(label_file: Path) -> bool:
    lines = label_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 1:
            continue
        cls = parts[0]
        if int(float(cls)) == 0:
            return True
    return False


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        p
        for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    labeled = []
    for img in images:
        label = args.labels_dir / f"{img.stem}.txt"
        if label.exists() and label.stat().st_size > 0 and has_class0(label):
            labeled.append(img)

    needed = args.rows * args.cols
    if len(labeled) < needed:
        raise ValueError(
            f"Not enough labeled images: need {needed}, found {len(labeled)} in {args.images_dir}"
        )

    selected = rng.sample(labeled, needed)
    font = get_font()

    canvas_w = args.cols * args.cell_size
    canvas_h = args.rows * args.cell_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (25, 25, 25))

    manifest_lines = []
    for idx, img_path in enumerate(selected):
        r, c = divmod(idx, args.cols)
        x0 = c * args.cell_size
        y0 = r * args.cell_size

        img = Image.open(img_path).convert("RGB").resize(
            (args.cell_size, args.cell_size), Image.Resampling.LANCZOS
        )
        label_file = args.labels_dir / f"{img_path.stem}.txt"
        tile = draw_yolo_boxes(img, label_file, font)
        canvas.paste(tile, (x0, y0))
        manifest_lines.append(str(img_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    args.manifest.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(f"Created image: {args.output}")
    print(f"Created manifest: {args.manifest}")


if __name__ == "__main__":
    main()
