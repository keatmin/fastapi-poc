from fastapi import FastAPI, Body
from depressionAI import DepressionAI
from typing import Dict
from pydantic import BaseModel


app = FastAPI()
learn = DepressionAI()


class ChatText(BaseModel):
    text: str


class Prediction(BaseModel):
    result: str
    probability: float


@app.post("/api/v1/predict", response_model=Prediction)
async def predict(req: ChatText) -> Dict[str,str]:
    return learn.predict(req.text)
