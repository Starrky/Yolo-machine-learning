import os
import shutil
import random
import yaml
from pathlib import Path
from pyaml_env import parse_config, BaseConfig


# Load config
config = BaseConfig(parse_config('../../configs/project_config.yaml'))
use_dataset = config.project_settings.set_dataset
dataset_location = "../../datasets"

# === CONFIG ===
DATASET_DIR = Path(f"{dataset_location}/{use_dataset}")     # original dataset folder
OUTPUT_DIR = Path(f"{dataset_location}/{use_dataset}-unpacked")  # temp folder for new split
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test
# ==============


def collect_all_image_paths(dataset_dir: Path):
    """Collect image paths from Train.txt, Test.txt, Validation.txt, etc."""
    txt_files = list(dataset_dir.glob("*.txt"))
    all_image_paths = []

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_image_paths.extend(lines)

    return list(set(all_image_paths))  # deduplicate if needed


def find_file_recursive(base_dir: Path, filename: str):
    """Find a file recursively under base_dir, return Path or None."""
    for p in base_dir.rglob(filename):
        return p
    return None


def main():
    data_yaml_file = DATASET_DIR / "data.yaml"

    # Load original classes
    with open(data_yaml_file, "r") as f:
        original_data = yaml.safe_load(f)
        class_names = original_data.get("names", [])
        num_classes = len(class_names)

    # Collect image paths from all available txt files
    image_paths = collect_all_image_paths(DATASET_DIR)
    random.shuffle(image_paths)

    n_total = len(image_paths)
    n_train = int(SPLIT_RATIO[0] * n_total)
    n_val = int(SPLIT_RATIO[1] * n_total)

    splits = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train + n_val],
        "test": image_paths[n_train + n_val:]
    }

    deleted_labels = 0
    deleted_images = 0

    cleanup_log_file = OUTPUT_DIR / "cleanup_log.txt"
    cleanup_log_file.parent.mkdir(parents=True, exist_ok=True)
    cleanup_lines = []

    for split_name, split_imgs in splits.items():
        split_dir = OUTPUT_DIR / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        txt_file = split_dir / f"{split_name}.txt"

        with open(txt_file, "w") as txt_out:
            for img_path in split_imgs:
                img_name = os.path.basename(img_path)
                lbl_name = os.path.splitext(img_name)[0] + ".txt"

                src_img = find_file_recursive(DATASET_DIR / "images", img_name)
                src_lbl = find_file_recursive(DATASET_DIR / "labels", lbl_name)

                dst_img = img_dir / img_name
                dst_lbl = lbl_dir / lbl_name

                try:
                    if src_img and src_img.exists():
                        shutil.copy(src_img, dst_img)
                    else:
                        raise FileNotFoundError(str(img_name))

                    if src_lbl and src_lbl.exists():
                        shutil.copy(src_lbl, dst_lbl)
                    else:
                        raise FileNotFoundError(str(lbl_name))

                    # Write relative path in txt
                    txt_out.write(f"images/{img_name}\n")

                except FileNotFoundError as e:
                    missing_path = str(e).lower()
                    if missing_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        if src_lbl and src_lbl.exists():
                            src_lbl.unlink()
                            deleted_labels += 1
                            msg = f"Deleted label: {src_lbl} because image was missing"
                            print(f"⚠️ {msg}")
                            cleanup_lines.append(msg)
                    elif missing_path.endswith(".txt"):
                        if src_img and src_img.exists():
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

    out_yaml = OUTPUT_DIR / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("\n✅ Dataset successfully split!")
    print(f"🧹 Deleted {deleted_images} orphan images and {deleted_labels} orphan labels.")
    print(f"📄 data.yaml written to: {out_yaml}")

    # Replace original folder
    try:
        shutil.rmtree(DATASET_DIR)
        OUTPUT_DIR.rename(DATASET_DIR)
        print(f"✅ Replaced original folder '{DATASET_DIR}' with unpacked dataset.")
    except Exception as e:
        print(f"❌ Error replacing original folder: {e}")


if __name__ == "__main__":
    main()
