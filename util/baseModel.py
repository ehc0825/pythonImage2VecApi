from pydantic import BaseModel



class UrlItem(BaseModel):
    image_url: str