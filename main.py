import random
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import VQAModel

from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
import torch.nn.functional as F
import os
import shutil
import uuid


from easy_vqa import get_test_image_paths, get_answers

# ---- Setup ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Ensure upload dir exists
os.makedirs("static/images", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Models and Metadata ----
vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

answers = get_answers()
answer2idx = {a: i for i, a in enumerate(answers)}
idx2answer = {i: a for i, a in enumerate(answers)}

model = VQAModel(vit_model, bert_model, hidden_dim=256, num_classes=len(answer2idx)).to(device)
model.load_state_dict(torch.load("vqa_vit_bert_model.pth", map_location=device))
model.eval()

# Test images
test_image_paths = get_test_image_paths()
max_question_len = 20

# ---- Inference ----
def vqa_inference_vit(image_path, question):
    image = Image.open(image_path).convert("RGB")
    extractor_input = vit_model.config.to_dict().get("image_size", 224)
    from transformers import AutoImageProcessor
    image_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_tensor = image_extractor(images=image, return_tensors="pt")["pixel_values"].to(device)

    encoded = tokenizer(question, padding="max_length", truncation=True, max_length=max_question_len, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        answer = idx2answer[pred_idx.item()]
        return answer, round(confidence.item(), 2)

# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, image_path: str = None):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_path": image_path,
        "answer": None
    })


from fastapi.responses import RedirectResponse

@app.get("/random-image/")
async def get_random_image():
    img_path = random.choice(test_image_paths)
    filename = os.path.basename(img_path)
    static_img_path = f"static/images/{filename}"
    shutil.copy(img_path, static_img_path)
    return RedirectResponse(url=f"/?image_path=/static/images/{filename}", status_code=302)


@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    question: str = Form(...),
    image_file: UploadFile = File(None),  
    image_path: str = Form(None),         
):
    # Case 1: Uploaded image
    if image_file is not None:
        contents = await image_file.read()
        save_path = f"static/images/{image_file.filename}"
        with open(save_path, "wb") as f:
            f.write(contents)
        used_image_path = save_path

    # Case 2: Image from hidden input
    elif image_path is not None:
        used_image_path = image_path

    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "image_path": None,
            "answer": "❌ No image provided.",
        })
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    local_image_path = os.path.join(BASE_DIR, used_image_path.lstrip("/"))
    # Run inference
    answer, confidence = vqa_inference_vit(local_image_path, question)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": used_image_path,
        "question": question,
        "answer": answer,
        "confidence": f"{confidence * 100:.2f}%"
    })