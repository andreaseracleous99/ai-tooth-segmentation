import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMG_DIR = Path("/datasets/train/11/images")
ANN_PATH = Path("/datasets/train/11/annotations_coco_bbox.json")

with ANN_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = {im["id"]: im for im in coco["images"]}
cats = {c["id"]: c["name"] for c in coco["categories"]}

img_id = coco["images"][0]["id"]
img_info = images[img_id]
img_path = IMG_DIR / img_info["file_name"]

anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

img = Image.open(img_path).convert("RGB")

fig, ax = plt.subplots()
ax.imshow(img)

for a in anns[:80]: 
    x, y, w, h = a["bbox"]
    rect = patches.Rectangle((x, y), w, h, fill=False, linewidth=2)
    ax.add_patch(rect)
    label = cats.get(a["category_id"], str(a["category_id"]))
    ax.text(x, y-3, label, fontsize=8)

ax.axis("off")
plt.show()

print("Image:", img_info["file_name"])
print("Annotations on image:", len(anns))
print("Example categories:", sorted(set([cats[a["category_id"]] for a in anns])))
