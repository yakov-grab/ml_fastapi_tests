from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    try:
        result = classifier(item.text)[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {e}")
