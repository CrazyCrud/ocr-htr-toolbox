import os
import argparse
from pathlib import Path


# See https://github.com/ultralytics/yolov5/issues/11337#issuecomment-1508645062
def seg_to_bbox(seg_info):
    class_id, *points = seg_info.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    bbox_info = f"{int(class_id)} {x_center} {y_center} {width} {height}"
    return bbox_info

def convert_annotations(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        bbox_line = seg_to_bbox(line.strip())
                        outfile.write(bbox_line + '\n')

    print(f"Conversion complete. Bounding box annotations saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation annotations to bounding box format.')
    parser.add_argument('--input_folder', '-i', required=True, help='Path to input folder containing segmentation annotations')
    parser.add_argument('--output_folder', '-o', required=True, help='Path to output folder for bounding box annotations')

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    convert_annotations(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()