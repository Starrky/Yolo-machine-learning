import json
import os
from urllib.parse import unquote
import shutil
import random
#####
##### Script used to prepare dataset for YOLOv8 training from json export in Label Studio
#####
# === CONFIG ===
PROJECT_NAME = ""
EXPORT_FILE = "../export.json"
IMAGES_DIR = r""
OUTPUT_DIR = r"{}/".format(PROJECT_NAME)


# Clear dir before preparing
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
print("Cleared the dir: {}".format(OUTPUT_DIR))

VAL_FRACTION = 0.1   # 10% validation
TEST_FRACTION = 0.1  # 10% test

# Create split folders with images/ and labels/
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Label mapping
label_map = {}
next_class_id = 0

# Load export
with open(EXPORT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle data for random split
random.shuffle(data)
n = len(data)
n_val = int(n * VAL_FRACTION)
n_test = int(n * TEST_FRACTION)

for i, task in enumerate(data):
    # Decode path
    image_url = task["data"]["image"]
    decoded_path = unquote(image_url)
    image_name = os.path.basename(decoded_path)
    src_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(src_path):
        print(f"⚠️ Skipping {image_name}, not found in {IMAGES_DIR}")
        continue

    # Determine split
    if i < n_test:
        split = "test"
    elif i < n_test + n_val:
        split = "val"
    else:
        split = "train"

    # Destination paths
    img_dst = os.path.join(OUTPUT_DIR, split, "images", image_name)
    label_dst = os.path.join(OUTPUT_DIR, split, "labels", os.path.splitext(image_name)[0] + ".txt")

    # Copy image
    shutil.copy2(src_path, img_dst)

    # Prepare YOLO annotation
    yolo_lines = []
    for ann in task.get("annotations", []):
        for result in ann.get("result", []):
            if result["type"] == "rectanglelabels":
                val = result["value"]
                label = val["rectanglelabels"][0] if val["rectanglelabels"] else "unknown"

                if label not in label_map:
                    label_map[label] = next_class_id
                    next_class_id += 1
                class_id = label_map[label]

                # Convert to YOLO normalized format
                x = val["x"] / 100
                y = val["y"] / 100
                w = val["width"] / 100
                h = val["height"] / 100
                x_center = x + w / 2
                y_center = y + h / 2

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    if yolo_lines:
        with open(label_dst, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(yolo_lines))

print("✅ Conversion complete!")
print("Classes:", label_map)

# === Create data.yaml ===
data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("test: test/images\n")
    f.write(f"nc: {len(label_map)}\n")
    # Store names as a Python-style list
    names_list = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    f.write(f"names: {names_list}\n")

print(f"✅ data.yaml created at {data_yaml_path}")


