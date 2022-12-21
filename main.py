from fastapi import FastAPI
from pydantic import BaseModel


class UrlItem(BaseModel):
    image_url: str

app = FastAPI()

@app.get("/urlImagevector")
async def converUrl(item: UrlItem):
        return {"msg":"hello World"}