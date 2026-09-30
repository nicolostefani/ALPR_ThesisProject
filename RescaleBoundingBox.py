"""
Rescale a CVAT annotations.xml from its original video resolution to a new resolution.

Usage:
    python scale_annotations.py input.xml output.xml --preset 4k
    python scale_annotations.py input.xml output.xml --preset 2k
"""

import argparse
import xml.etree.ElementTree as ET

PRESETS = {
    "4k": (3840, 2160),
    "2k": (2560, 1440),  # QHD (2K)
}


def parse_args():
    p = argparse.ArgumentParser(description="Rescale CVAT box annotations to a new video resolution.")
    p.add_argument("input", help="Path to the source annotations.xml")
    p.add_argument("output", help="Path where the rescaled XML will be written")
    p.add_argument("--preset", choices=PRESETS.keys(), help="Target resolution preset")
    p.add_argument("--width", type=int, help="Target width in pixels (overrides preset)")
    p.add_argument("--height", type=int, help="Target height in pixels (overrides preset)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.width and args.height:
        target_w, target_h = args.width, args.height
    elif args.preset:
        target_w, target_h = PRESETS[args.preset]
    else:
        raise SystemExit("Specify either --preset {4k,2k} or both --width and --height")

    tree = ET.parse(args.input)
    root = tree.getroot()

    # Build per-task scale factors based on the original size of each task
    task_scale = {}
    for task in root.iter("task"):
        task_id = task.findtext("id")
        size_el = task.find("original_size")
        if size_el is None or task_id is None:
            continue
        src_w = float(size_el.findtext("width"))
        src_h = float(size_el.findtext("height"))
        task_scale[task_id] = (target_w / src_w, target_h / src_h)
        size_el.find("width").text = str(target_w)
        size_el.find("height").text = str(target_h)

    # Scale every box using the scale factor of the task it belongs to
    n_boxes = 0
    for track in root.iter("track"):
        task_id = track.get("task_id")
        scale_x, scale_y = task_scale.get(task_id, (1.0, 1.0))
        for box in track.findall("box"):
            box.set("xtl", f'{float(box.get("xtl")) * scale_x:.2f}')
            box.set("xbr", f'{float(box.get("xbr")) * scale_x:.2f}')
            box.set("ytl", f'{float(box.get("ytl")) * scale_y:.2f}')
            box.set("ybr", f'{float(box.get("ybr")) * scale_y:.2f}')
            n_boxes += 1

    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Saved {n_boxes} rescaled boxes to {args.output} ({target_w}x{target_h})")


if __name__ == "__main__":
    main()