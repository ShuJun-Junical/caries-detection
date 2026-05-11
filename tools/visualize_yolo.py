#!/usr/bin/env python3
"""
Visualize YOLO-format labels. Draws boxes for multiple classes with distinct colors.
Usage examples:
    python tools/visualize_yolo.py --labels /tmp/class1_files.txt --classes 1 --outdir runs/vis_class1 --max 8
    python tools/visualize_yolo.py --labels /tmp/both.txt --classes 0,1 --outdir runs/vis_compare_both --max 20
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def yolo_to_boxes(lbl_path, img_w, img_h):
    boxes = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def find_image_for_label(lbl_path):
    p = Path(lbl_path)
    # Common patterns: same folder sibling in ../images or replacing /yolo/ with /images
    candidates = []
    # sibling with various extensions
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif'):
        cand = p.with_suffix(ext)
        candidates.append(cand)
    # try replacing /yolo/ with /images/
    parts = p.parts
    try:
        idx = parts.index('yolo')
        newparts = list(parts)
        newparts[idx] = 'images'
        imgbase = Path(*newparts).with_suffix('')
        for ext in ('.jpg', '.jpeg', '.png'):
            candidates.append(imgbase.with_suffix(ext))
    except ValueError:
        pass
    # also try same basename under dataset/*/images
    for root in ('dataset/train/images', 'dataset/valid/images', 'dataset/test/images'):
        candidates.append(Path(root) / (p.stem + '.jpg'))
        candidates.append(Path(root) / (p.stem + '.png'))
    for c in candidates:
        if c.exists():
            return c
    return None


def draw_boxes(img_path, boxes, class_filters, colors, out_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for cls, (x1,y1,x2,y2) in boxes:
        if cls not in class_filters:
            continue
        color = colors.get(cls, 'red')
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        label = str(cls)
        if font:
            draw.text((x1+4, y1+4), label, fill=color, font=font)
        else:
            draw.text((x1+4, y1+4), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True, help='file containing label paths, one per line')
    p.add_argument('--classes', required=True, help='comma-separated class ids to draw, e.g. 0,1')
    p.add_argument('--colors', default='', help='comma-separated colors matching classes, e.g. red,blue')
    p.add_argument('--outdir', default='runs/vis_class', help='output directory')
    p.add_argument('--max', type=int, default=8)
    args = p.parse_args()
    lbls = [line.strip() for line in open(args.labels, 'r') if line.strip()]
    class_filters = [int(x) for x in args.classes.split(',') if x.strip()!='']
    color_list = [c for c in args.colors.split(',') if c.strip()!=''] if args.colors else []
    colors = {}
    for i,cls in enumerate(class_filters):
        if i < len(color_list):
            colors[cls] = color_list[i]
        else:
            # default palette
            colors[cls] = 'red' if cls==0 else 'blue'

    count = 0
    for lbl in lbls:
        if count >= args.max:
            break
        lblp = Path(lbl)
        imgp = find_image_for_label(lblp)
        if imgp is None:
            continue
        with Image.open(imgp) as im:
            iw, ih = im.size
        boxes = yolo_to_boxes(lblp, iw, ih)
        # skip if no target class in this file
        if not any(c in class_filters for c,_ in boxes):
            continue
        suffix = '_compare' if len(class_filters) > 1 else f'_class{class_filters[0]}'
        outpath = Path(args.outdir) / (lblp.stem + suffix + '.jpg')
        draw_boxes(imgp, boxes, class_filters, colors, outpath)
        count += 1
    print(f'Wrote {count} images to {args.outdir}')

if __name__ == '__main__':
    main()
