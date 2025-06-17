import textToSpeech as tts
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pdf2image import convert_from_path
from typing import List
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

app = FastAPI()

# Define frontend URL for CORS
origins = [
    "http://localhost:5173",
]

# CORS middleware configuration for the FastAPI app to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "jpeg", "jpg", "txt"}
colour_map = {
        "protanopia": 0,
        "deuteranopia": 1,
        "tritanopia": 2,
    }

# Check if the file has an allowed extension
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', -1)[-1].lower() in ALLOWED_EXTENSIONS

# Endpoint to upload a file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Clear previous uploads
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Clear previous conversions
    shutil.rmtree(CONVERTED_FOLDER, ignore_errors=True)
    os.makedirs(CONVERTED_FOLDER, exist_ok=True)

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename}

# Pydantic model for conversion types
class ConversionTypes(BaseModel):
    toSpeech: bool
    toColour: bool
    colour: str

def colour_to_int(colour: str) -> int:
    return colour_map.get(colour.lower(), -1)  

async def process_pdf_to_pdf(input_path: str, output_path: str, colour_val: int):
    original_pages = []

    # Convert PDF to images
    pages = convert_from_path(input_path, dpi=850)

    # Save each page as a JPEG image
    for count, page in enumerate(pages):
        page_path = f"{output_path}_{count + 1}.jpg"
        page.save(page_path, "JPEG")
        original_pages.append(page_path)

    # Process each image with the model
    processed_pages = []
    for path in original_pages:
        processed_image = run_inference(path, colour_val, "model_training/cnn_model.pth19")
        processed_pages.append(processed_image)
    
    # Save processed images as a single PDF
    processed_pages[0].save(output_path, save_all=True, append_images=processed_pages[1:])

    # Clean up original images
    for path in original_pages:
        os.remove(path)

async def convert_file(input_path: str, output_path: str, toSpeech: bool, toColour: bool, colour: str):
    ext = input_path.rsplit('.', -1)[-1].lower()
    colour_val = colour_to_int(colour)
    if ext == "txt":
        # Convert text file to speech
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tts.text_to_speech(text, output_path)
        return
    
    if ext in {"jpeg", "jpg"}:
        # Convert image to colour blindness corrected image
        image = run_inference(input_path, colour_val, "model_training/cnn_model.pth19")
        image.save(output_path, "JPEG")
        return
    
    if toColour:
        # Convert PDF to JPG and then run inference
        await process_pdf_to_pdf(input_path, output_path, colour_val)
    if toSpeech:
        # Convert PDF to text and then to speech
        text = tts.extract_selectable(input_path)
        tts.text_to_speech(text, output_path)

@app.post("/convert/")
async def convert(data: ConversionTypes):
    if not data.toSpeech and not data.toColour:
        raise HTTPException(status_code=400, detail="At least one conversion must be selected")

    converted_files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_ext = filename.rsplit('.', 1)[-1].lower()
        file_base, _ = os.path.splitext(filename)
        
        converted_path = ""
        if file_ext == "pdf":
            if data.toColour:
                converted_path = os.path.join(CONVERTED_FOLDER, f"{file_base}_converted.pdf")
                await convert_file(file_path, converted_path, False, data.toColour, data.colour)
                converted_files.append(converted_path)
            if data.toSpeech:
                converted_path = os.path.join(CONVERTED_FOLDER, f"{file_base}_corrected.wav")
                await convert_file(file_path, converted_path, data.toSpeech, False, data.colour)
                converted_files.append(converted_path)
        elif file_ext in {"jpeg", "jpg"}:
            converted_path = os.path.join(CONVERTED_FOLDER, f"{file_base}_corrected.jpg")
            await convert_file(file_path, converted_path, data.toSpeech, data.toColour, data.colour)
            converted_files.append(converted_path)
        else:
            converted_path = os.path.join(CONVERTED_FOLDER, f"{file_base}_converted.wav")
            await convert_file(file_path, converted_path, data.toSpeech, data.toColour, data.colour)
            converted_files.append(converted_path)
    return {"message": "Files converted successfully", "converted_files": [os.path.basename(f) for f in converted_files]}

@app.get("/download/")
async def download_file():
    if os.path.exists(CONVERTED_FOLDER):
        zip_path = "converted_files.zip"
        shutil.make_archive("converted_files", 'zip', CONVERTED_FOLDER)
        return FileResponse(zip_path, media_type='application/zip', filename='converted_files.zip')

class ColorBlindnessCNN(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1), 
            nn.Sigmoid()  
        )

    def forward(self, x, labels):
        batch_size, _, h, w = x.size()
        label_onehot = F.one_hot(labels, num_classes=3).float() 
        label_maps = label_onehot.view(batch_size, 3, 1, 1).expand(-1, -1, h, w)
        
        x = torch.cat([x, label_maps], dim=1)  
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def run_inference(image_path: str, class_label: int, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = ColorBlindnessCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024))
    ])

    # Load and transform input image
    img = Image.open(image_path).convert("RGB")
    size = img.size
    img_tensor = transform(img).unsqueeze(0).to(device) # add batch dimension

    # Prepare label tensor
    label_tensor = torch.tensor([class_label], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor, label_tensor)

    # Convert output to PIL image
    output_img = TF.to_pil_image(output.squeeze(0).cpu())
    output_img = output_img.resize(size, Image.BICUBIC)

    # Save output image
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # output_img.save(output_path)
    # print(f"Saved corrected image to {output_path}")
    return output_img
