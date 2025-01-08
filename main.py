from SentiScope.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from SentiScope.pipeline.data_profiler_pipeline import DataProfilerPipeline
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
     
