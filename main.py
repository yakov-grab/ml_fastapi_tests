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
        raise HTTPException(status_code=500,
                            detail=f"Error processing text: {e}")


@app.post("/translate/")
def translate(item: Item, target_language: str):
    translation_pipeline = pipeline("translation",
                                    model="Helsinki-NLP/opus-mt-en-xx")
    return translation_pipeline(item.text,
                                target_language=target_language)


@app.post("/keywords/")
def keywords(item: Item):
    keywords_pipeline = pipeline("keyword-extraction")
    return keywords_pipeline(item.text)
