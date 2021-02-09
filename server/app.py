from tensorflow.keras.preprocessing.image import img_to_array, load_img
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

import requests

templates = Jinja2Templates(directory='server/templates')

async def preprocess_image_to_array(file: File, target_size=(256, 256)):
    content = await file.read()
    
    with open('server/static/processed_image.jpg', 'wb') as _f:
        _f.write(content)

    image = load_img(
        "server/static/processed_image.jpg", target_size=target_size, 
    )

    arr = img_to_array(image) / 255
    return np.expand_dims(arr, 0)

app = FastAPI()

app.mount("/static/", StaticFiles(directory="server/static/"), name="static")

@app.post("/process_xray/")
async def create_upload_file(file: UploadFile = File(...)):
    arr = await preprocess_image_to_array(file)

    data = {'instances': arr.tolist()}
    res = requests.post('http://tensorflow_serving:8501/v1/models/xray:predict', json=data)

    return res.json()

@app.get("/", response_class=HTMLResponse)
async def show_index(request: Request):
    return templates.TemplateResponse("index.html", context={'request': request})