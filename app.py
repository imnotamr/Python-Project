from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import Dict
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
import skimage.io
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

app = FastAPI()

os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

weights = 'densenet121-res224-all'
resize = True
cuda = torch.cuda.is_available()
model = xrv.models.get_model(weights)
if cuda:
    model = model.cuda()

# Disease info dictionary
disease_info = {
    "Atelectasis": {
        "description": "Collapse or incomplete expansion of part of the lung.",
        "recommendation": "Encourage deep breathing and physiotherapy. Monitor oxygen levels.",
        "treatment": "Address underlying cause, possibly bronchoscopy or chest physiotherapy.",
        "reason": "The image shows loss of lung volume with increased opacity in a specific region, suggesting collapse."
    },
    "Cardiomegaly": {
        "description": "Enlargement of the heart, possibly indicating heart disease.",
        "recommendation": "Refer to cardiologist. Perform echocardiogram.",
        "treatment": "Depends on cause: medications like beta blockers or ACE inhibitors.",
        "reason": "The heart silhouette appears disproportionately large relative to thoracic width."
    },
    "Effusion": {
        "description": "Fluid accumulation in the pleural space around the lungs.",
        "recommendation": "Investigate cause. Consider drainage if symptomatic.",
        "treatment": "Thoracentesis, diuretics, or treat underlying condition.",
        "reason": "Blunting of costophrenic angles and layering of fluid is visible on X-ray."
    },
    "Infiltration": {
        "description": "Presence of abnormal substances in lung tissue, often indicating infection or inflammation.",
        "recommendation": "Follow-up with clinical correlation and lab tests.",
        "treatment": "Antibiotics if infection confirmed. Monitor progression.",
        "reason": "Diffuse or patchy opacities present throughout the lung fields."
    },
    "Mass": {
        "description": "A well-defined opacity suggesting a solid lesion or tumor.",
        "recommendation": "CT scan and possibly biopsy. Refer to oncology.",
        "treatment": "Surgical resection or oncological treatment based on diagnosis.",
        "reason": "A solitary, dense, round shadow seen in lung zone."
    },
    "Nodule": {
        "description": "A small, round opacity usually <3cm. May be benign or malignant.",
        "recommendation": "Compare with previous imaging. Consider follow-up CT.",
        "treatment": "Monitor or biopsy depending on risk factors.",
        "reason": "Small, localized, round opacity identified in the lung parenchyma."
    },
    "Pneumonia": {
        "description": "Infection in lung tissue causing consolidation.",
        "recommendation": "Start antibiotics. Monitor temperature and oxygenation.",
        "treatment": "Antibiotic therapy tailored to infection.",
        "reason": "Lobar or segmental consolidation pattern seen with air bronchograms."
    },
    "Pneumothorax": {
        "description": "Air in pleural space causing lung collapse.",
        "recommendation": "Immediate evaluation. Chest tube if significant.",
        "treatment": "Needle decompression or chest tube placement.",
        "reason": "Visible absence of lung markings in peripheral zone with sharp pleural edge."
    },
    "Consolidation": {
        "description": "Solidification of lung tissue due to fluid or infection.",
        "recommendation": "Evaluate for pneumonia. Clinical correlation required.",
        "treatment": "Antibiotics or anti-inflammatory depending on cause.",
        "reason": "Homogenous opacity in lung field with obscured vascular markings."
    },
    "Edema": {
        "description": "Fluid buildup in lung tissue, commonly due to heart failure.",
        "recommendation": "Monitor vitals. Administer diuretics if needed.",
        "treatment": "Diuretics and treatment of underlying heart failure.",
        "reason": "Bilateral haziness and perihilar opacities indicating fluid overload."
    },
    "Emphysema": {
        "description": "Air trapping and destruction of lung alveoli, often from smoking.",
        "recommendation": "Encourage smoking cessation. Pulmonary function testing.",
        "treatment": "Bronchodilators, steroids, and pulmonary rehab.",
        "reason": "Hyperinflated lungs with flattened diaphragm and reduced vascular markings."
    },
    "Fibrosis": {
        "description": "Scarring of lung tissue, reducing elasticity.",
        "recommendation": "High-resolution CT and pulmonary evaluation.",
        "treatment": "Steroids, antifibrotics, or lung transplant in severe cases.",
        "reason": "Reticular markings and volume loss in lung bases."
    },
    "Pleural_Thickening": {
        "description": "Fibrotic thickening of pleura due to chronic inflammation or asbestos.",
        "recommendation": "Check history of exposure. CT for extent.",
        "treatment": "Usually observation unless symptomatic.",
        "reason": "Irregular pleural margins or thickened pleural lining visible."
    },
    "Hernia": {
        "description": "Protrusion of abdominal content into thoracic cavity.",
        "recommendation": "Refer to surgery. Confirm with CT scan.",
        "treatment": "Surgical repair if symptomatic or large.",
        "reason": "Gas-filled bowel loops seen above diaphragm level."
    },
    "Lung Lesion": {
        "description": "Abnormal tissue or damage in the lung.",
        "recommendation": "Further imaging with CT or PET scan.",
        "treatment": "Biopsy if suspicious. Monitor or treat accordingly.",
        "reason": "Focal irregularity or shadow detected on X-ray."
    },
    "Fracture": {
        "description": "Break or crack in rib bones often due to trauma.",
        "recommendation": "Pain management and monitor for pneumothorax.",
        "treatment": "Rest and analgesics. Rarely requires surgical fixation.",
        "reason": "Disruption in the continuity of the rib shadow on X-ray."
    },
    "Lung Opacity": {
        "description": "Partial loss of lung transparency due to fluid, cells, or other materials.",
        "recommendation": "Correlate clinically. Chest CT if needed.",
        "treatment": "Depends on cause — antibiotics, diuretics, or other therapies.",
        "reason": "The X-ray shows hazy or opaque areas indicating potential pathology."
    },
    "Enlarged Cardiomediastinum": {
        "description": "Widening of mediastinum which may indicate pathology such as hemorrhage or mass.",
        "recommendation": "Urgent evaluation with CT if trauma or suspicion of aortic injury.",
        "treatment": "Treat underlying cause (e.g., surgery for dissection).",
        "reason": "Increased width of mediastinal shadow observed on X-ray."
    }
}



def predict_image(image: np.ndarray) -> Dict[str, float]:
    image = xrv.datasets.normalize(image, 255)
    if len(image.shape) > 2:
        image = image[:, :, 0]
    image = image[None, :, :]
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224) if resize else xrv.datasets.XRayCenterCrop()
    ])
    image = transform(image)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    if cuda:
        image_tensor = image_tensor.cuda()
    with torch.no_grad():
        preds = model(image_tensor).cpu()
        return {k: float(v) for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())}

# API Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = skimage.io.imread(io.BytesIO(await file.read()))
    predictions = predict_image(image)
    pred_label = max(predictions, key=predictions.get)
    pred_score = predictions[pred_label]

    # Grad-CAM
    if len(image.shape) > 2:
        image = image[:, :, 0]
    image = xrv.datasets.normalize(image, 255)
    image = image[None, :, :]
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    image = transform(image)
    input_tensor = torch.from_numpy(image).unsqueeze(0).requires_grad_(True)
    if cuda:
        input_tensor = input_tensor.cuda()

    features, gradients = [], []

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
    model.zero_grad()
    output[0, xrv.datasets.default_pathologies.index(pred_label)].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = features[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    img_base = image[0]
    cam_img = Image.fromarray(cam).resize((img_base.shape[1], img_base.shape[0]))
    cam_img = np.array(cam_img)

    report_id = str(uuid.uuid4())
    cam_filename = f"images/cam_{report_id}.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(img_base, cmap='gray')
    plt.imshow(cam_img, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(cam_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    info = disease_info.get(pred_label, {
        "description": "No description available.",
        "recommendation": "No recommendation available.",
        "treatment": "No treatment info available.",
        "reason": "No reason available."
    })

    report = {
        "report_id": report_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "diagnosis": pred_label,
        "confidence": round(pred_score, 3),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "treatment": info["treatment"],
        "reason": info["reason"],
        "heatmap_image_url": f"http://127.0.0.1:8000/{cam_filename}"
    }

    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)