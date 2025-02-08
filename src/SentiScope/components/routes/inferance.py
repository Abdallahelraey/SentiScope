from fastapi import FastAPI, APIRouter, Depends, HTTPException
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from SentiScope.pipeline.base_line_infereance_pipeline import BaseLineInferancePipeline
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.logging import logger
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, UploadFile, Form, File
import pandas as pd
import numpy as np
import uvicorn
import io
import sys
import os
import mlflow


# Configuration Management
config = ConfigurationManager()
mlflow_config = config.get_mlflow_config() 

# Configuration for inference
MODEL_NAME = "logistic_regression"
STAGE = "Production"
VECTORIZER_PATH = r"artifacts\feature_transformation\20250122_122840\bow_vectorizer.joblib"
LABEL_ENCODER_PATH = r"artifacts\feature_transformation\20250122_122840\label_encoder.joblib"

# Global MLflow Tracker
global_mlflow_tracker = None

# Create FastAPI app and router
app = FastAPI()

sentiment_router = APIRouter(
    prefix="/api/v1/sentiment",
    tags=["api_v1", "sentiment"],
)

class PredictionRequest(BaseModel):
    texts: List[str]

def get_mlflow_tracker():
    """
    Ensure MLflow tracker is initialized.
    If not, create and initialize a new one.
    """
    global global_mlflow_tracker
    
    # If tracker doesn't exist, create it
    if global_mlflow_tracker is None:
        try:
            # Create MLflow tracker
            global_mlflow_tracker = MLflowTracker(config=mlflow_config)
            
            # Start a run for the entire application
            global_mlflow_tracker.start_run("FastAPIInference")
            
            # Log additional context about the inference service
            global_mlflow_tracker.log_params({
                "service_type": "sentiment_inference",
                "model_name": MODEL_NAME,
                "model_stage": STAGE
            })
            
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {str(e)}")
            raise RuntimeError(f"MLflow initialization failed: {str(e)}")
    
    return global_mlflow_tracker

# Startup event to initialize MLflow tracking
@app.on_event("startup")
async def startup_event():
    """
    Ensure MLflow tracking is initialized when the FastAPI application starts.
    """
    try:
        get_mlflow_tracker()
    except Exception as e:
        logger.error(f"Startup MLflow initialization failed: {str(e)}")

# Shutdown event to cleanup MLflow tracking
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup MLflow tracking when the FastAPI application stops.
    """
    global global_mlflow_tracker
    if global_mlflow_tracker:
        try:
            global_mlflow_tracker.end_run()
            logger.info("MLflow run ended for FastAPI application")
        except Exception as e:
            logger.error(f"Error during MLflow cleanup: {str(e)}")
        finally:
            global_mlflow_tracker = None

# Prediction endpoint
@sentiment_router.post("/predict")
async def predict_route(request: PredictionRequest):  
    try:
        # Get the global MLflow tracker (creates one if not exists)
        mlflow_tracker = get_mlflow_tracker()
        
        # Log input details
        mlflow_tracker.log_params({
            "input_text_count": len(request.texts)
        })
        
        # Log the start of prediction
        logger.info(f">>>>>>> Sentiment Prediction Started <<<<<<<")
        
        # Initialize the baseline inference pipeline
        baseline_modeling = BaseLineInferancePipeline(mlflow_tracker=mlflow_tracker)
        
        # Perform prediction
        predictions = baseline_modeling.main(
            model_name=MODEL_NAME, 
            stage=STAGE, 
            data=request.texts, 
            vectorizer_path=VECTORIZER_PATH, 
            label_encoder_path=LABEL_ENCODER_PATH
        )

        predictions = list(predictions) if not isinstance(predictions, list) else predictions
        # Log completion of prediction
        logger.info(f">>>>>> Sentiment Prediction Completed <<<<<<<")
        
        # Return predictions
        return {
            "predictions": predictions,
            "input_texts": request.texts
        }
    
    except Exception as e:
        # Log the exception
        logger.exception(f"Error during sentiment prediction: {str(e)}")
        
        # Log error metrics
        get_mlflow_tracker().log_metrics({
            "prediction_error": 1.0,
            "error_message": str(e)
        })
        
        # Raise HTTP exception with details
        raise HTTPException(status_code=500, detail=str(e))



dataframe_sentiment_router = APIRouter(
    prefix="/api/v1/sentiment",
    tags=["api_v1", "sentiment"]
)

class DataFramePredictionRequest(BaseModel):
    """
    Pydantic model for DataFrame prediction request
    Allows specifying a column to analyze or using the whole DataFrame
    """
    dataframe: dict  # Allows receiving DataFrame as a dictionary
    text_column: Optional[str] = None  # Optional column name for text analysis



@dataframe_sentiment_router.post("/predict_csv")
async def predict_csv_route(
    file: UploadFile = File(...), 
    text_column: str = Form(...),
):
    try:
        # --- 1. Read and Validate CSV ---
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty CSV file")
            
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{text_column}' not found in CSV headers"
            )

        # --- 2. Clean and Validate Texts ---
        # Handle missing values and convert to strings
        df[text_column] = df[text_column].fillna('').astype(str)
        texts = df[text_column].tolist()
        
        if not texts or all(text.strip() == '' for text in texts):
            raise HTTPException(
                status_code=400,
                detail="No valid text data found in selected column"
            )

        # --- 3. Get Predictions ---
        baseline_modeling = BaseLineInferancePipeline(
            mlflow_tracker=get_mlflow_tracker()
        )
        
        predictions = baseline_modeling.main(
            model_name=MODEL_NAME,
            stage=STAGE,
            data=texts,
            vectorizer_path=VECTORIZER_PATH,
            label_encoder_path=LABEL_ENCODER_PATH
        )

        # --- 4. Validate Predictions ---
        predictions = np.array(predictions)
        
        if predictions.size == 0 or len(predictions) != len(texts):
            raise ValueError(
                "Mismatch between input texts and predictions. "
                f"Received {len(texts)} texts but {len(predictions)} predictions."
            )

        # --- 5. Prepare Results ---
        result_df = df.copy()
        result_df['predicted_sentiment'] = predictions
        
        # Convert NaN/None to empty strings for JSON safety
        result_df = result_df.fillna('')

        return {
            "dataframe": result_df.to_dict(orient='records'),
            "stats": {
                "total_texts": len(texts),
                "predictions_count": len(predictions),
                "sample_predictions": list(predictions[:3])  # Convert to list for JSON serialization
            }
        }

    except HTTPException as he:
        # Re-raise existing HTTP exceptions
        raise he
        
    except Exception as e:
        logger.exception(f"CSV prediction failed: {str(e)}")
        get_mlflow_tracker().log_metrics({"prediction_error": 1.0})
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Application creation function
def create_app():
    app = FastAPI()
    app.include_router(sentiment_router)
    app.include_router(dataframe_sentiment_router)
    return app

# Main entry point for running the server
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
# Run the server and test the endpoint using the following JSON payload:

'''
{
    "texts": [
        "This is a new text to predict that is good", 
        "Another example text that is bad"
    ]
}
'''