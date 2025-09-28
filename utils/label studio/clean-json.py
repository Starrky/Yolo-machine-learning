"""
Script to clean JSON export from Label Studio. Remove mismatches:
if the image doesn't have a label or if the label doesn't have an image

Need to provide the following variables:
- EXPORT_FILE: path to the JSON export file from Label Studio
- OUTPUT_FILE: path to the cleaned JSON export file you want to use later
- IMAGES_DIR: path to the directory where the images are stored that were used for the label studio locally
"""

import json
import os
from urllib.parse import unquote

# === CONFIG ===
EXPORT_FILE = "export.json"
OUTPUT_FILE = "export_cleaned.json"
IMAGES_DIR = r""
TXT_OUTPUT_FILE = "../removed.json"

# Load JSON
with open(EXPORT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned_data = []
missing_files = []

for task in data:
    if "data" in task and "image" in task["data"]:
        image_url = task["data"]["image"]
        decoded = unquote(image_url)

        # Remove prefix `/data/local-files/?d=`
        if "?d=" in decoded:
            decoded = decoded.split("?d=")[-1]

        filename = os.path.basename(decoded)
        local_path = os.path.join(IMAGES_DIR, filename)

        if os.path.exists(local_path):
            cleaned_data.append(task)
        else:
            missing_files.append(filename)

# Save cleaned JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

# Save cleaned JSON
with open(TXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(missing_files, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned JSON saved to {OUTPUT_FILE}")
print(f"👉 {len(cleaned_data)} tasks kept")
print(f"❌ {len(missing_files)} tasks removed (missing images)")
