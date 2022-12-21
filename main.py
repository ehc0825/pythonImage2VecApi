from fastapi import FastAPI
from pydantic import BaseModel
from img2vec_pytorch import Img2Vec


class UrlItem(BaseModel):
    image_url: str

app = FastAPI()

@app.get("/urlImagevector")
async def converUrl(item: UrlItem):
        return {"msg":"hello World"}