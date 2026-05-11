#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-dataset}"

for split in train valid test; do
    img_dir="$DATASET_DIR/$split/images"
    yolo_dir="$DATASET_DIR/$split/yolo"

    if [[ ! -d "$img_dir" ]]; then
        echo "[$split] images目录不存在: $img_dir"
        continue
    fi

    if [[ ! -d "$yolo_dir" ]]; then
        echo "[$split] yolo目录不存在: $yolo_dir"
        continue
    fi

    img_count=$(find "$img_dir" -type f \( \
        -iname "*.jpg" -o \
        -iname "*.jpeg" -o \
        -iname "*.png" -o \
        -iname "*.bmp" -o \
        -iname "*.webp" \
    \) | wc -l | tr -d ' ')

    label_count=$(find "$yolo_dir" -type f -iname "*.txt" | wc -l | tr -d ' ')

    echo "[$split]"
    echo "  images: $img_count"
    echo "  labels: $label_count"

    if [[ "$img_count" -eq "$label_count" ]]; then
        echo "  status: OK，数量一致"
    else
        echo "  status: ERROR，数量不一致"
    fi

    echo
done
