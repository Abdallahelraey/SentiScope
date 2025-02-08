from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from SentiScope.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from SentiScope.pipeline.data_profiler_pipeline import DataProfilerPipeline
from SentiScope.pipeline.data_transformation_pipeline import DataTransformerPipeline
from SentiScope.pipeline.baseline_modeling_pipeline import BaseLineModelingPipeline
from SentiScope.pipeline.base_line_infereance_pipeline import BaseLineInferancePipeline
from SentiScope.pipeline.advanced_modeling_pipeline import TransformerModelPipeline
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger

# Define default arguments for Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 8),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    "senti_scope_pipeline",
    default_args=default_args,
    description="Automating SentiScope NLP Workflow using Airflow",
    schedule_interval="@daily",
    catchup=False,
)

# Initialize Configuration & MLflow Tracker
def setup_mlflow():
    global main_mlflow_tracker
    config = ConfigurationManager()
    mlflow_config = config.get_mlflow_config()
    main_mlflow_tracker = MLflowTracker(config=mlflow_config)
    main_mlflow_tracker.start_run("MainPipeline")

def end_mlflow():
    main_mlflow_tracker.end_run()

# Define Task Functions
def run_data_ingestion():
    logger.info("Starting Data Ingestion...")
    ingestion = DataIngestionPipeline(mlflow_tracker=main_mlflow_tracker)
    ingestion.main()
    logger.info("Data Ingestion Completed.")

def run_data_profiling():
    logger.info("Starting Data Profiling...")
    profiler = DataProfilerPipeline(mlflow_tracker=main_mlflow_tracker)
    profiler.main()
    logger.info("Data Profiling Completed.")

def run_data_transformation():
    logger.info("Starting Data Transformation...")
    transformer = DataTransformerPipeline(mlflow_tracker=main_mlflow_tracker)
    transformer.main()
    logger.info("Data Transformation Completed.")

def run_baseline_modeling():
    logger.info("Starting Baseline Modeling...")
    baseline_model = BaseLineModelingPipeline(mlflow_tracker=main_mlflow_tracker)
    baseline_model.main()
    logger.info("Baseline Modeling Completed.")

def run_advanced_modeling():
    logger.info("Starting Advanced Modeling...")
    transformer_model = TransformerModelPipeline(mlflow_tracker=main_mlflow_tracker)
    transformer_model.main()
    logger.info("Advanced Modeling Completed.")

def run_baseline_inference():
    logger.info("Starting Baseline Inference...")
    model_name = "logistic_regression"
    stage = "Production"
    X_test = ["This is a new text to predict that is good", "Another example text that is bad"]
    vectorizer_path = r"artifacts\feature_transformation\20250122_122840\bow_vectorizer.joblib"
    label_encoder_path = r"artifacts\feature_transformation\20250122_122840\label_encoder.joblib"
    inference = BaseLineInferancePipeline(mlflow_tracker=main_mlflow_tracker)
    inference.main(model_name=model_name, stage=stage, data=X_test, vectorizer_path=vectorizer_path, label_encoder_path=label_encoder_path)
    logger.info("Baseline Inference Completed.")

# Define Airflow Tasks
setup_task = PythonOperator(
    task_id="setup_mlflow",
    python_callable=setup_mlflow,
    dag=dag,
)

data_ingestion_task = PythonOperator(
    task_id="data_ingestion",
    python_callable=run_data_ingestion,
    dag=dag,
)

data_profiling_task = PythonOperator(
    task_id="data_profiling",
    python_callable=run_data_profiling,
    dag=dag,
)

data_transformation_task = PythonOperator(
    task_id="data_transformation",
    python_callable=run_data_transformation,
    dag=dag,
)

baseline_modeling_task = PythonOperator(
    task_id="baseline_modeling",
    python_callable=run_baseline_modeling,
    dag=dag,
)

advanced_modeling_task = PythonOperator(
    task_id="advanced_modeling",
    python_callable=run_advanced_modeling,
    dag=dag,
)

baseline_inference_task = PythonOperator(
    task_id="baseline_inference",
    python_callable=run_baseline_inference,
    dag=dag,
)

end_task = PythonOperator(
    task_id="end_mlflow",
    python_callable=end_mlflow,
    dag=dag,
)

# Define Task Dependencies
setup_task >> data_ingestion_task >> data_profiling_task >> data_transformation_task 
data_transformation_task >> baseline_modeling_task >> advanced_modeling_task >> baseline_inference_task >> end_task
