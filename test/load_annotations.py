import json
from pathlib import Path

ANN_PATH = Path("/datasets/train/11/annotations_coco_bbox.json")

with ANN_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

print("✅ COCO loaded")
print("Images:", len(coco.get("images", [])))
print("Annotations:", len(coco.get("annotations", [])))
print("Categories:", len(coco.get("categories", [])))

print("\nFirst image entry:\n", coco["images"][0])
print("\nFirst annotation entry:\n", coco["annotations"][0])
print("\nCategories (id -> name):")
for c in coco["categories"][:20]:
    print(c["id"], "->", c["name"])
