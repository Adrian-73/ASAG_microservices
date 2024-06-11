from fastapi import FastAPI
import Embeddings
from pydantic import BaseModel
evaluter = Embeddings.evaluate
app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


class QApair(BaseModel):
    question:str
    key:str
    answer:str

class ResponseSheet(BaseModel):
    pairs : dict


@app.get("/evaluate")
def evaluate(req : QApair):
    return round(evaluter(req.question,req.key,req.answer),2)

@app.get("/evaluate_paper")
def evaluate(req: ResponseSheet):
    res = dict()
    qa = req.pairs
    for i in qa.keys():
       res[i] = round(evaluter(qa[i]["question"],qa[i]["key"],qa[i]["answer"]),2)

    return res
