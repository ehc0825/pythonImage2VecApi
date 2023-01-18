from fastapi import FastAPI
from pydantic import BaseModel
from img2vec_pytorch import Img2Vec
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
import requests
import json
import numpy as np
from util import jsonParser
from util import baseModel
from util import imageParser

UrlItem=baseModel.UrlItem
app = FastAPI()

@app.post("/urlImagevector")
async def converUrl(item: UrlItem):
    urlItem=dict(item)
    image_url=(str(urlItem['image_url']))
    requestImage=requests.get(image_url)
    img2Vec=Img2Vec(cuda=False);
    image=Image.open(BytesIO(requestImage.content)).convert('RGB')
    imageVector=img2Vec.get_vec(image)
    return {"vector":imageVector.tolist()}


@app.post("/urlImageLabel")
async def urlImageLabel(item: UrlItem):
    dicted_item=dict(item)
    item_path=str(dicted_item['image_url'])
    model = ResNet50(weights='imagenet')
    model.summary()
    res = requests.get(item_path)
    img = load_img(BytesIO(res.content), target_size=(224, 224))
    imageProcessor=imageParser.ImageParser
    modelInput = imageProcessor.imgToModelInput(img)
    preds = model.predict(modelInput)
    label = decode_predictions(preds, top=3)[0]
    returnjson = json.dumps(label,cls=jsonParser.NumpyEncoder)
    return{"label":returnjson}

