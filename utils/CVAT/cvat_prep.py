import os
import shutil
import random
import yaml
from pathlib import Path
from pyaml_env import parse_config, BaseConfig


# Load config
config = BaseConfig(parse_config('../../configs/project_config.yaml'))
use_dataset = config.project_settings.set_dataset

# === CONFIG ===
DATASET_DIR = use_dataset        # original dataset folder
OUTPUT_DIR = f"{use_dataset}-unpacked"   # temp folder for new split
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test
# ==============

def main():
    images_dir = Path(DATASET_DIR) / "images/Train"
    labels_dir = Path(DATASET_DIR) / "labels/Train"
    train_file = Path(DATASET_DIR) / "Train.txt"
    data_yaml_file = Path(DATASET_DIR) / "data.yaml"

    # Load original classes
    with open(data_yaml_file, "r") as f:
        original_data = yaml.safe_load(f)
        class_names = original_data.get("names", [])
        num_classes = len(class_names)

    # Read image paths
    with open(train_file, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    random.shuffle(image_paths)

    n_total = len(image_paths)
    n_train = int(SPLIT_RATIO[0] * n_total)
    n_val = int(SPLIT_RATIO[1] * n_total)

    splits = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train+n_val],
        "test": image_paths[n_train+n_val:]
    }

    deleted_labels = 0
    deleted_images = 0

    # Prepare cleanup log
    cleanup_log_file = Path(OUTPUT_DIR) / "cleanup_log.txt"
    cleanup_log_file.parent.mkdir(parents=True, exist_ok=True)
    cleanup_lines = []

    for split_name, split_imgs in splits.items():
        split_dir = Path(OUTPUT_DIR) / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        txt_file = split_dir / f"{split_name}.txt"

        with open(txt_file, "w") as txt_out:
            for img_path in split_imgs:
                img_name = os.path.basename(img_path)
                lbl_name = os.path.splitext(img_name)[0] + ".txt"

                src_img = images_dir / img_name
                src_lbl = labels_dir / lbl_name

                dst_img = img_dir / img_name
                dst_lbl = lbl_dir / lbl_name

                try:
                    # Copy image
                    if src_img.exists():
                        shutil.copy(src_img, dst_img)
                    else:
                        raise FileNotFoundError(str(src_img))

                    # Copy label
                    if src_lbl.exists():
                        shutil.copy(src_lbl, dst_lbl)
                    else:
                        raise FileNotFoundError(str(src_lbl))

                    # Write relative path in txt
                    txt_out.write(f"images/{img_name}\n")

                except FileNotFoundError as e:
                    missing_path = str(e).lower()
                    if missing_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        if src_lbl.exists():
                            src_lbl.unlink()
                            deleted_labels += 1
                            msg = f"Deleted label: {src_lbl} because image was missing"
                            print(f"⚠️ {msg}")
                            cleanup_lines.append(msg)
                    elif missing_path.endswith(".txt"):
                        if src_img.exists():
                            src_img.unlink()
                            deleted_images += 1
                            msg = f"Deleted image: {src_img} because label was missing"
                            print(f"⚠️ {msg}")
                            cleanup_lines.append(msg)
                    else:
                        print(f"⚠️ Unhandled missing file: {missing_path}")
                        cleanup_lines.append(f"Unhandled missing file: {missing_path}")

    # Write cleanup log
    if cleanup_lines:
        with open(cleanup_log_file, "w") as log_f:
            log_f.write("\n".join(cleanup_lines))
        print(f"📝 Cleanup log saved to: {cleanup_log_file}")

    # Create new data.yaml
    yaml_data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names
    }

    out_yaml = Path(OUTPUT_DIR) / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("\n✅ Dataset successfully split!")
    print(f"🧹 Deleted {deleted_images} orphan images and {deleted_labels} orphan labels.")
    print(f"📄 data.yaml written to: {out_yaml}")

    # Replace original folder
    try:
        shutil.rmtree(DATASET_DIR)
        Path(OUTPUT_DIR).rename(DATASET_DIR)
        print(f"✅ Replaced original folder '{DATASET_DIR}' with unpacked dataset.")
    except Exception as e:
        print(f"❌ Error replacing original folder: {e}")

if __name__ == "__main__":
    main()
