#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

LABELME_DIR = Path("./dataset/train/labelme")
IMAGE_DIR = Path("./dataset/train/images")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def image_stem_from_json(data, json_path):
    image_path = data.get("imagePath")
    if image_path:
        return Path(image_path).stem
    return json_path.stem

def main():
    json_files = sorted(LABELME_DIR.glob("*.json"))

    total_json = 0
    json_with_shapes = 0
    json_without_shapes = 0
    total_shapes = 0

    label_counter = Counter()
    shape_type_counter = Counter()
    shapes_per_json_counter = Counter()

    bad_json_files = []
    missing_required_fields = []
    bad_shapes = []
    bad_points = []
    bad_rectangles = []

    json_stems = set()
    json_stems_with_shapes = set()
    json_stems_without_shapes = set()

    for jp in json_files:
        total_json += 1

        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            bad_json_files.append((jp, str(e)))
            continue

        stem = image_stem_from_json(data, jp)
        json_stems.add(stem)

        for field in ["imagePath", "imageHeight", "imageWidth", "shapes"]:
            if field not in data:
                missing_required_fields.append((jp, field))

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            bad_shapes.append((jp, "shapes is not a list"))
            shapes = []

        n_shapes = len(shapes)
        total_shapes += n_shapes
        shapes_per_json_counter[n_shapes] += 1

        if n_shapes > 0:
            json_with_shapes += 1
            json_stems_with_shapes.add(stem)
        else:
            json_without_shapes += 1
            json_stems_without_shapes.add(stem)

        for idx, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                bad_shapes.append((jp, f"shape[{idx}] is not a dict"))
                continue

            label = shape.get("label")
            shape_type = shape.get("shape_type")
            points = shape.get("points")

            label_counter[label] += 1
            shape_type_counter[shape_type] += 1

            if not label:
                bad_shapes.append((jp, f"shape[{idx}] missing label"))

            if not shape_type:
                bad_shapes.append((jp, f"shape[{idx}] missing shape_type"))

            if not isinstance(points, list) or len(points) == 0:
                bad_points.append((jp, idx, "points missing or empty"))
                continue

            for p in points:
                if (
                    not isinstance(p, list)
                    or len(p) != 2
                    or not all(isinstance(x, (int, float)) for x in p)
                ):
                    bad_points.append((jp, idx, f"bad point: {p}"))

            # 如果是 rectangle，通常应该有两个点：左上/右下，或者两对角点
            if shape_type == "rectangle":
                if len(points) != 2:
                    bad_rectangles.append((jp, idx, f"rectangle has {len(points)} points"))
                else:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    if x1 == x2 or y1 == y2:
                        bad_rectangles.append((jp, idx, f"zero-area rectangle: {points}"))

    print("========== LabelMe Summary ==========")
    print(f"labelme dir:               {LABELME_DIR}")
    print(f"json files:                {total_json}")
    print(f"json with shapes:          {json_with_shapes}")
    print(f"json without shapes:       {json_without_shapes}")
    print(f"total shapes:              {total_shapes}")

    print()
    print("========== Labels ==========")
    for label, count in label_counter.most_common():
        print(f"{str(label):<30} {count}")

    print()
    print("========== Shape Types ==========")
    for shape_type, count in shape_type_counter.most_common():
        print(f"{str(shape_type):<20} {count}")

    print()
    print("========== Shapes Per JSON ==========")
    for n_shapes, n_files in sorted(shapes_per_json_counter.items()):
        print(f"{n_shapes:>4} shape(s): {n_files} json file(s)")

    print()
    print("========== Basic Problems ==========")
    print(f"bad json files:            {len(bad_json_files)}")
    print(f"missing required fields:   {len(missing_required_fields)}")
    print(f"bad shapes:                {len(bad_shapes)}")
    print(f"bad points:                {len(bad_points)}")
    print(f"bad rectangles:            {len(bad_rectangles)}")

    if bad_json_files:
        print()
        print("First 20 bad json files:")
        for jp, err in bad_json_files[:20]:
            print(f"{jp}: {err}")

    if missing_required_fields:
        print()
        print("First 20 missing required fields:")
        for jp, field in missing_required_fields[:20]:
            print(f"{jp}: missing {field}")

    if bad_points:
        print()
        print("First 20 bad points:")
        for jp, idx, msg in bad_points[:20]:
            print(f"{jp}: shape[{idx}] {msg}")

    if bad_rectangles:
        print()
        print("First 20 bad rectangles:")
        for jp, idx, msg in bad_rectangles[:20]:
            print(f"{jp}: shape[{idx}] {msg}")

    if IMAGE_DIR.exists():
        image_stems = {
            p.stem for p in IMAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }

        images_without_json = image_stems - json_stems
        json_without_image = json_stems - image_stems

        print()
        print("========== LabelMe JSON vs Disk Images ==========")
        print(f"image files on disk:       {len(image_stems)}")
        print(f"json stems:                {len(json_stems)}")
        print(f"images without json:       {len(images_without_json)}")
        print(f"json without image:        {len(json_without_image)}")

        if images_without_json:
            print()
            print("First 50 images without LabelMe json:")
            for stem in sorted(images_without_json)[:50]:
                print(stem)

        if json_without_image:
            print()
            print("First 50 LabelMe json without image:")
            for stem in sorted(json_without_image)[:50]:
                print(stem)

    if json_stems_without_shapes:
        print()
        print("========== First 50 LabelMe JSON Without Shapes ==========")
        for stem in sorted(json_stems_without_shapes)[:50]:
            print(stem)

if __name__ == "__main__":
    main()