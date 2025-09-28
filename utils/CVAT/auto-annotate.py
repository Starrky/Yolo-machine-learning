#!/usr/bin/env python
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
from pyaml_env import parse_config, BaseConfig
from cvat_sdk import make_client
import cv2
import numpy as np
import imagehash
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time

# Optional imports
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ----------------- USER SETTINGS -----------------
config = BaseConfig(parse_config('../../configs/project_config.yaml'))

CVAT_URL = config.CVAT.url
USERNAME = config.CVAT.username
PASSWORD = config.CVAT.password

detection_model = f"../../models/{config.autolabel.detection_model}"
segmentation_model = f"../../models/{config.autolabel.segmentation_model}"
SAM_model = config.autolabel.sam_model

use_folder = config.autolabel.label_folder     #   <-------

project_path = Path(f"../../autolabel/{use_folder}")
data_path = project_path / "images"
labels_dir = project_path / "labels"
zip_output = project_path / use_folder
processed_folder = project_path / "processed_vids"
dup_dir = project_path / "found_duplicates"
frames_root = data_path
video_folder = data_path

# Ensure directories exist
for folder in [processed_folder, labels_dir, dup_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# ----------------- HELPER FUNCTIONS -----------------
def build_cvat_index(client):
    """Build a dict: filename -> (project, task, frame_index)"""
    index = {}
    for project in client.projects.list():
        project_id = project.id
        try:
            project_tasks, _ = client.api_client.tasks_api.list(project_id=project_id)
        except Exception as e:
            print(f"⚠️ Could not list tasks for project {project_id}: {e}")
            continue
        for task in project_tasks.get("results", []):
            task_id = task["id"]
            try:
                data_meta, _ = client.api_client.tasks_api.retrieve_data_meta(task_id)
            except Exception as e:
                print(f"⚠️ Could not retrieve data meta for task {task_id}: {e}")
                continue
            for idx, frame in enumerate(data_meta.get("frames", [])):
                fname = Path(frame.get("name", "")).name
                index[fname] = {
                    "project": project_id,
                    "task": task_id,
                    "frame_index": idx,
                    "frame_name": frame.get("name", "")
                }
    return index

def is_significant_hash(frame_cv, kept_hashes, threshold=5):
    """Check uniqueness using perceptual hash (fast)."""
    pil_img = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    h = imagehash.phash(pil_img)
    for kh in kept_hashes:
        if h - kh <= threshold:
            return False
    kept_hashes.add(h)
    return True

def extract_strict_unique_frames(video_path, cvat_index, similarity_threshold=5, stride_high_fps=10, stride_low_fps=5):
    """Extract strict, unique frames from video or GIF using phash (fast)."""
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    kept_hashes = set()
    saved_count = 0
    existing_frames = set(cvat_index.keys())
    is_gif = video_path.lower().endswith(".gif")

    if is_gif:
        img = Image.open(video_path)
        total_frames = getattr(img, "n_frames", 1)
        stride = stride_high_fps
        frame_range = range(total_frames)
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        stride = stride_high_fps if fps >= 60 else stride_low_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_range = range(0, total_frames, stride)

    print(f"[{video_name}] Extracting frames...")
    for i in tqdm(frame_range, desc=f"Processing {video_name}", unit="frame"):
        if is_gif:
            img.seek(i)
            frame = img.convert("RGB")
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame_cv = cap.read()
            if not ret:
                continue

        frame_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
        if frame_filename in existing_frames:
            continue

        if is_significant_hash(frame_cv, kept_hashes, threshold=similarity_threshold):
            cv2.imwrite(os.path.join(video_dir, frame_filename), frame_cv)
            saved_count += 1

    if not is_gif:
        cap.release()
    print(f"[{video_name}] Extracted {saved_count} strict unique frames.")
    return saved_count

def _prettify_xml(elem: ET.Element) -> str:
    raw = ET.tostring(elem, encoding='utf-8')
    parsed = minidom.parseString(raw)
    return parsed.toprettyxml(indent='  ', encoding='utf-8')

# ----------------- MAIN SCRIPT -----------------
if __name__ == "__main__":
    # Build CVAT index
    print("[*] Building CVAT filename index...")
    with make_client(CVAT_URL) as client:
        client.login((USERNAME, PASSWORD))
        cvat_index = build_cvat_index(client)

    # Process videos/GIFs
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".gif")
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(video_folder, filename)
            extract_strict_unique_frames(video_path, cvat_index, similarity_threshold=5)
            shutil.move(video_path, os.path.join(processed_folder, filename))
            print(f"Moved {filename} to {processed_folder}")

    # Check duplicates after extraction
    file_list = [f for f in os.listdir(data_path) if (data_path / f).is_file()]
    moved_count, kept_count = 0, 0
    for fname in tqdm(file_list, desc="Checking duplicates against CVAT", unit="img"):
        if fname in cvat_index:
            shutil.move(str(data_path / fname), dup_dir / fname)
            moved_count += 1
        else:
            kept_count += 1
    print(f"[*] {kept_count} new images kept, {moved_count} duplicates moved to {dup_dir}\n")

    # ----------------- MANUAL REVIEW PAUSE -----------------
    print("[*] Frame extraction complete.")
    time.sleep(3)
    manual_review = input("Do you want to manually review/delete images before annotation? [y/N]: ").strip().lower()
    if manual_review == 'y':
        # Open folder in Windows Explorer
        os.startfile(data_path)
        while True:
            cont = input("Type 'continue' when you are done reviewing/deleting images: ").strip().lower()
            if cont == 'continue':
                break

    # Re-scan the images folder after manual review
    file_list = [f for f in os.listdir(data_path) if (data_path / f).is_file()]
    print(f"[*] {len(file_list)} images found after manual review.\n")

    # Auto-annotate remaining frames
    print("[*] Auto-annotating…")
    auto_annotate(
        data=data_path,
        det_model=segmentation_model,
        sam_model=SAM_model,
        output_dir=labels_dir,
        conf=0.45,
        iou=0.45,
        device="cuda"
    )
    print("[*] Annotation finished.\n")


    def remove_empty_annotations(image_dir, labels_dir):
        """
        Removes images that auto_annotate did not find anything for
        (i.e., no .txt exists or .txt is empty).
        """
        removed = 0
        kept = 0

        for img_file in Path(image_dir).glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            stem = img_file.stem
            label_txt = labels_dir / f"{stem}.txt"

            # Case 1: No label file
            if not label_txt.exists():
                img_file.unlink()
                removed += 1
                continue

            # Case 2: Empty file (no detections)
            if label_txt.stat().st_size == 0:
                img_file.unlink()
                label_txt.unlink()  # also remove the empty label
                removed += 1
                continue

            kept += 1

        print(f"[*] Removed {removed} images without annotations, kept {kept} images.")


    # Cleanup: remove images with no annotations
    remove_empty_annotations(data_path, labels_dir)

    # Load detection model
    model = YOLO(detection_model)
    class_names = list(model.names.values())
    num_classes  = len(class_names)
    print(f"[*] Model has {num_classes} classes: {class_names}\n")

    # Gather images
    image_files = sorted([p for p in data_path.glob("*.*") if p.is_file()])
    print(f"[*] Found {len(image_files)} images in {data_path}\n")

    # Build CVAT 1.1 annotations.xml
    print("[*] Building annotations.xml (CVAT 1.1)...")
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task_el = ET.SubElement(meta, 'task')
    ET.SubElement(task_el, 'name').text = use_folder
    ET.SubElement(task_el, 'size').text = str(len(image_files))
    labels_el = ET.SubElement(task_el, 'labels')
    for cname in class_names:
        label_el = ET.SubElement(labels_el, 'label')
        ET.SubElement(label_el, 'name').text = str(cname)
        ET.SubElement(label_el, 'attributes')

    # Populate annotations (YOLO + mask polygons)
    for idx, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"⚠️ Could not read image {img_path}: {e} -- skipping")
            continue
        image_el = ET.SubElement(annotations, 'image', id=str(idx), name=img_path.name, width=str(w), height=str(h))
        label_txt = labels_dir / f"{img_path.stem}.txt"
        if label_txt.exists():
            with open(label_txt, 'r') as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except:
                        continue
                    if len(parts) > 5:
                        poly_vals = list(map(float, parts[5:]))
                        if len(poly_vals) % 2 != 0:
                            continue
                        pts = [f"{poly_vals[i]*w:.2f},{poly_vals[i+1]*h:.2f}" if max(poly_vals)<=1.001 else f"{poly_vals[i]:.2f},{poly_vals[i+1]:.2f}" for i in range(0,len(poly_vals),2)]
                        points_str = ';'.join(pts)
                        ET.SubElement(image_el, 'polygon', label=str(class_names[cls_id]), occluded='0', source='auto', points=points_str)

        mask_png = labels_dir / f"{img_path.stem}.png"
        if mask_png.exists() and _HAS_CV2:
            try:
                mask = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape)==3 else mask
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c)<10: continue
                    pts = [f"{float(p[0]):.2f},{float(p[1]):.2f}" for p in c.reshape(-1,2)]
                    points_str = ';'.join(pts)
                    label_name = class_names[0] if class_names else 'object'
                    ET.SubElement(image_el, 'polygon', label=str(label_name), occluded='0', source='auto', points=points_str)
            except: pass

    annotations_xml = project_path / 'annotations.xml'
    with open(annotations_xml, 'wb') as f:
        f.write(_prettify_xml(annotations))
    print(f"[*] annotations.xml written to {annotations_xml}\n")

    # Build CVAT export folder and zip
    print("[*] Building CVAT export folder and zipping...")
    cvat_tmp = project_path / 'cvat_export'
    if cvat_tmp.exists(): shutil.rmtree(cvat_tmp)
    cvat_tmp.mkdir(parents=True)
    img_dir = cvat_tmp / 'images'
    img_dir.mkdir()
    for img_path in image_files:
        shutil.copy2(img_path, img_dir / img_path.name)
    shutil.copy2(annotations_xml, cvat_tmp / 'annotations.xml')
    shutil.make_archive(str(zip_output), 'zip', str(cvat_tmp))
    print(f"[*] CVAT 1.1 archive created: {zip_output}.zip")
    print("\n✅ Finished – ready to import into CVAT (images/ + annotations.xml).")
