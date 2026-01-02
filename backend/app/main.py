from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from app.model import predict_image

app = FastAPI(title="Kanye Gesture API")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    gesture = predict_image(image)
    return {"gesture": gesture}
