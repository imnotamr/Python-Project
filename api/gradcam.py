import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
import skimage.io
from PIL import Image
import io
import base64



model = xrv.models.get_model("densenet121-res224-all")
model.eval()

image_path = "test_image.png"  
img = skimage.io.imread(image_path)

if len(img.shape) == 3:
    img = img[:, :, 0]  

img = xrv.datasets.normalize(img, 255)
img = img[None, :, :]  
transform = transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img = transform(img)

input_tensor = torch.from_numpy(img).unsqueeze(0).requires_grad_(True)

features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

try:
    target_layer = model.model.denseblock4
except AttributeError:
    target_layer = model.features.denseblock4

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

output = model(input_tensor)
pred_class_idx = output[0].argmax().item()
pred_label = xrv.datasets.default_pathologies[pred_class_idx]

model.zero_grad()
output[0, pred_class_idx].backward()

grads = gradients[0].detach().numpy()[0]
acts = features[0].detach().numpy()[0]
weights = np.mean(grads, axis=(1, 2))

cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i, :, :]

cam = np.maximum(cam, 0)
cam = cam / np.max(cam)
cam = Image.fromarray(cam).resize((img.shape[2], img.shape[1]))
cam = np.array(cam)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img[0], cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img[0], cmap="gray")
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM: {pred_label}")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save and encode image
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode("utf-8")
buf.close()

# Print it so you can use it in HTML
print(f"data:image/png;base64,{img_base64}")
