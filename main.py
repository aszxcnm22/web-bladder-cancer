import os
import uuid
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from ultralytics import YOLO
import cv2

# === FastAPI setup ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# === Load models ===
yolo_model = YOLO("Model/best.pt")  # ตรวจจับก้อนเนื้อ
vgg19_path = os.path.join("Model/vgg19_web_model.pt")
class_names = ["T1", "T2", "T3", "T4"]

# โหลดโมเดลเต็ม (full model)
vgg19 = torch.load(vgg19_path, map_location="cpu")
vgg19.eval()

# Transform สำหรับ VGG19
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Detection + Classification ===
def detect_and_predict(image_path, result_path):
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        orig_image = image.copy()

        results = yolo_model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        predicted_labels = []
        colors = [tuple(np.random.randint(0,255,3).tolist()) for _ in range(len(boxes))]

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            roi_pil = Image.fromarray(roi).resize((224,224))
            roi_tensor = transform(roi_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = vgg19(roi_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
                predicted_class = class_names[pred.item()]
                predicted_labels.append((predicted_class, float(confidence)))

            # แปลง RGB -> BGR สำหรับ OpenCV
            color_bgr = (colors[idx][2], colors[idx][1], colors[idx][0])
            label_text = f"{predicted_class} {confidence.item():.2f}"
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(orig_image, label_text, (x1, max(0,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        cv2.imwrite(result_path, cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR))
        return predicted_labels

    except Exception as e:
        print("Error in detect_and_predict:", e)
        return []

# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse ("index.html", {"request": request})

@app.get("/index.html", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact.html", response_class=HTMLResponse)
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.post("/contact.html", response_class=HTMLResponse)
async def process_contact(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Path for result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        # Detect + classify
        predictions = detect_and_predict(upload_path, result_path)

        # HTML fragment สำหรับ AJAX
        pred_html = f"""
        <div class="text-center">
            <h4>Uploaded Image</h4>
            <img src="/static/uploads/{filename}" class="preview-image">
        </div>
        <div class="text-center mt-3">
            <h4>Result Image</h4>
            <img src="/static/results/{result_filename}" class="preview-image">
        </div>
        <div class="mt-3">
            <h5>Predictions:</h5>
            <ul class="label-list">
        """
        if predictions:
            for cls, conf in predictions:
                pred_html += f"<li>{cls} — Confidence: {conf:.2f}</li>"
        else:
            pred_html += "<li>No detection</li>"
        pred_html += "</ul></div>"

        return HTMLResponse(content=pred_html)

    except Exception as e:
        print("Error in process_contact:", e)
        return HTMLResponse(content="<p style='color:red;'>เกิดข้อผิดพลาดในการประมวลผลไฟล์</p>")
