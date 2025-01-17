from SentiScope.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from SentiScope.pipeline.data_profiler_pipeline import DataProfilerPipeline
from SentiScope.pipeline.data_transformation_pipeline import DataTransformerPipeline
from SentiScope.pipeline.baseline_modeling_pipeline import BaseLineModelingPipeline
from SentiScope.pipeline.advanced_modeling_pipeline import TransformerModelPipeline
from SentiScope.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
    
    
    
STAGE_NAME = "Data Profiling stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_Profiler = DataProfilerPipeline()
   data_Profiler.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_transformer = DataTransformerPipeline()
   data_transformer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
     
     
STAGE_NAME = "Modeling Baseline stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   baseline_modeling = BaseLineModelingPipeline()
   baseline_modeling.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Advanced Modeling stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   transformer_model = TransformerModelPipeline()
   transformer_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e