from fastapi import FastAPI
from pydantic import BaseModel
from img2vec_pytorch import Img2Vec
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests


class UrlItem(BaseModel):
    image_url: str

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