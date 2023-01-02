from fastapi import FastAPI
from pydantic import BaseModel
from img2vec_pytorch import Img2Vec
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
import json
import numpy as np
from util import jsonParser
from util import baseModel

UrlItem=baseModel.UrlItem
app = FastAPI()

@app.get("/urlImagevector")
async def converUrl(item: UrlItem):
    urlItem=dict(item)
    image_url=(str(urlItem['image_url']))
    requestImage=requests.get(image_url)
    img2Vec=Img2Vec(cuda=False);
    image=Image.open(BytesIO(requestImage.content)).convert('RGB')
    imageVector=img2Vec.get_vec(image)
    return {"vector":imageVector.tolist()}


@app.get("/urlImageLabel")
async def urlImageLabel(item: UrlItem):
    dicted_item=dict(item)
    item_path=str(dicted_item['image_url'])
    model = ResNet50(weights='imagenet')
    model.summary()
    res = requests.get(item_path)
    img = load_img(BytesIO(res.content), target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    label = decode_predictions(preds, top=3)[0]
    returnjson = json.dumps(label,cls=jsonParser.NumpyEncoder)
    return{"label":returnjson}
