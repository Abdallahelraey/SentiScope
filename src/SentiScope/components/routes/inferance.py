from fastapi import FastAPI, APIRouter, Depends, HTTPException
import uvicorn
import sys
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from SentiScope.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

sentiment_router = APIRouter(
    prefix="/api/v1/sentiment",
    tags=["api_v1", "sentiment"],
)

@sentiment_router.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")




@sentiment_router.post("/predict")
async def predict_route(text: str):  
    try:
        pass  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

def create_app():
    app = FastAPI()
    app.include_router(sentiment_router)
    return app
    
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)