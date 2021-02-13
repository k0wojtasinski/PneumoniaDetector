from tensorflow.keras.preprocessing.image import img_to_array, load_img
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

import requests

templates = Jinja2Templates(directory="server/templates")


async def preprocess_image_to_array(file: File, target_size=(256, 256)) -> np.array:
    """ helper function to preprocess given image to be suited for Tensorflow models.
    It also resizes image to 256 x 256, as the input of both models
    It saves images to server/static/processed_image to make it available for frontend

    Args:
        file (File): uploaded file
        target_size (tuple, optional): final resolution of image. Defaults to (256, 256).

    Returns:
        np.array: array to be used by Tensorflow model
    """
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

# static directory
app.mount("/static/", StaticFiles(directory="server/static/"), name="static")


@app.post("/process_xray/{model_type}")
async def create_upload_file(model_type: str, file: UploadFile = File(...)):
    """ route to process image based on given model_type
        it supports two model_types: xray and transferlearning
        it returns prediction of pneumonia based on provided image

    Args:
        model_type (str): type of model, which determines Tensorflow model
        file (UploadFile, optional): [description]. Defaults to File(...).

    Raises:
        HTTPException: when given model_type is not available

    Returns:
        float: prediction of pneumonia
    """
    if model_type not in ("xray", "transferlearning"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{model_type} model type is not available",
        )

    arr = await preprocess_image_to_array(file)

    data = {"instances": arr.tolist()}

    # we are making call to local TensorFlow Serving service to make prediction
    res = requests.post(
        f"http://tensorflow_serving:8501/v1/models/{model_type}:predict", json=data
    )

    predictions = res.json()["predictions"]

    return predictions[0][-1]


@app.get("/", response_class=HTMLResponse)
async def show_index(request: Request):
    """ route to show GUI for this application

    Args:
        request (Request): request

    Returns:
        HTMLResponse: static index.html file
    """
    return templates.TemplateResponse("index.html", context={"request": request})
