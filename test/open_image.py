from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/datasets/train/11/images/1.jpg").convert("RGB")

plt.imshow(img)
plt.axis("off")
plt.title("Panoramic X-ray")
plt.show()