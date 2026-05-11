#!/usr/bin/env python3
"""
遍历给定目录（默认当前工作目录），把所有标注中的类别 1 直接改为 0。
处理格式：YOLO .txt、COCO .json、LabelMe .json、Pascal VOC .xml。
谨慎提示：脚本会直接就地覆盖文件（按用户要求不做备份）。
用法：python tools/merge_class1_to_0.py /path/to/dataset
"""
import sys
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def process_yolo(path: Path):
    changed = 0
    for p in path.rglob('*.txt'):
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        lines = text.splitlines()
        new_lines = []
        file_changed = False
        for ln in lines:
            if not ln.strip():
                new_lines.append(ln)
                continue
            parts = ln.split()
            if parts[0] == '1':
                parts[0] = '0'
                file_changed = True
            new_lines.append(' '.join(parts))
        if file_changed:
            p.write_text('\n'.join(new_lines) + ('\n' if text.endswith('\n') else ''), encoding='utf-8')
            changed += 1
    return changed


def process_coco_json(p: Path):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return 0
    if not isinstance(data, dict) or 'annotations' not in data:
        return 0
    changed = 0
    for ann in data.get('annotations', []):
        if ann.get('category_id') == 1:
            ann['category_id'] = 0
            changed += 1
    if changed:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return 1 if changed else 0


def process_labelme_json(p: Path):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return 0
    if not isinstance(data, dict) or 'shapes' not in data:
        return 0
    file_changed = False
    for shape in data.get('shapes', []):
        lbl = shape.get('label')
        # 如果 label 是数字字符串或者数字 1，则改为 '0'
        if lbl == '1' or lbl == 1:
            shape['label'] = '0'
            file_changed = True
    if file_changed:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return 1
    return 0


def process_pascal_xml(p: Path):
    try:
        tree = ET.parse(p)
    except Exception:
        return 0
    root = tree.getroot()
    file_changed = False
    for obj in root.findall('object'):
        name = obj.find('name')
        if name is not None and name.text == '1':
            name.text = '0'
            file_changed = True
    if file_changed:
        tree.write(p, encoding='utf-8')
        return 1
    return 0


def main(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        print(f'路径不存在: {root_dir}')
        return 2

    stats = {
        'yolo_files_changed': 0,
        'coco_files_changed': 0,
        'labelme_files_changed': 0,
        'pascal_files_changed': 0,
    }

    # YOLO: 修改所有 .txt（注意：这也会改其他非标注的 txt，请谨慎）
    stats['yolo_files_changed'] = process_yolo(root)

    # JSON: 区分 COCO 和 LabelMe
    for p in root.rglob('*.json'):
        # 优先尝试 COCO（含 annotations 字段）
        try:
            text = p.read_text(encoding='utf-8')
            js = json.loads(text)
        except Exception:
            continue
        if isinstance(js, dict) and 'annotations' in js:
            stats['coco_files_changed'] += process_coco_json(p)
        elif isinstance(js, dict) and 'shapes' in js:
            stats['labelme_files_changed'] += process_labelme_json(p)

    # Pascal VOC XML
    for p in root.rglob('*.xml'):
        stats['pascal_files_changed'] += process_pascal_xml(p)

    print('=== 合并类别 1->0 完成 ===')
    print(f"YOLO 文本文件已修改数量: {stats['yolo_files_changed']}")
    print(f"COCO JSON 文件中含被改注释的文件数: {stats['coco_files_changed']}")
    print(f"LabelMe JSON 文件被修改数量: {stats['labelme_files_changed']}")
    print(f"Pascal VOC XML 文件被修改数量: {stats['pascal_files_changed']}")
    return 0


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    sys.exit(main(root))
