from SentiScope.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from SentiScope.pipeline.data_profiler_pipeline import DataProfilerPipeline
from SentiScope.pipeline.data_transformation_pipeline import DataTransformerPipeline
from SentiScope.pipeline.baseline_modeling_pipeline import BaseLineModelingPipeline
from SentiScope.pipeline.advanced_modeling_pipeline import TransformerModelPipeline
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger

config = ConfigurationManager()
mlflow_config = config.get_mlflow_config() 
main_mlflow_tracker = MLflowTracker(config=mlflow_config)
main_mlflow_tracker.start_run("MainPipeline")


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_ingestion = DataIngestionPipeline(mlflow_tracker=main_mlflow_tracker)
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
    
    
STAGE_NAME = "Data Profiling stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_Profiler = DataProfilerPipeline(mlflow_tracker=main_mlflow_tracker)
   data_Profiler.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_transformer = DataTransformerPipeline(mlflow_tracker=main_mlflow_tracker)
   data_transformer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
     
STAGE_NAME = "Modeling Baseline stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   baseline_modeling = BaseLineModelingPipeline(mlflow_tracker=main_mlflow_tracker)
   baseline_modeling.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Advanced Modeling stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   transformer_model = TransformerModelPipeline(mlflow_tracker=main_mlflow_tracker)
   transformer_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e

main_mlflow_tracker.end_run()