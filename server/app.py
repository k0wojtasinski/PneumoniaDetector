from tensorflow.keras.preprocessing.image import img_to_array, load_img
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

import requests

templates = Jinja2Templates(directory="server/templates")


async def preprocess_image_to_array(file: File, target_size=(256, 256)):
    content = await file.read()

    with open("server/static/processed_image.jpg", "wb") as _f:
        _f.write(content)

    image = load_img(
        "server/static/processed_image.jpg",
        target_size=target_size,
    )

    arr = img_to_array(image) / 255
    return np.expand_dims(arr, 0)


app = FastAPI()

app.mount("/static/", StaticFiles(directory="server/static/"), name="static")


@app.post("/process_xray/{model_type}")
async def create_upload_file(model_type: str, file: UploadFile = File(...)):
    if model_type not in ("xray", "transferlearning"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{model_type} model type is not available",
        )

    arr = await preprocess_image_to_array(file)

    data = {"instances": arr.tolist()}
    res = requests.post(
        f"http://tensorflow_serving:8501/v1/models/{model_type}:predict", json=data
    )

    predictions = res.json()["predictions"]

    return predictions[0][-1]


@app.get("/", response_class=HTMLResponse)
async def show_index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})
