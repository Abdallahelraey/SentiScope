import argparse
from SentiScope.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from SentiScope.pipeline.data_profiler_pipeline import DataProfilerPipeline 
from SentiScope.pipeline.data_transformation_pipeline import DataTransformerPipeline
from SentiScope.pipeline.baseline_modeling_pipeline import BaseLineModelingPipeline
from SentiScope.pipeline.base_line_infereance_pipeline import BaseLineInferancePipeline
from SentiScope.pipeline.advanced_modeling_pipeline import TransformerModelPipeline
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run specified stages of the pipeline.')
    valid_stages = [
        "Data Ingestion stage",
        "Data Profiling stage",
        "Data Transformation stage",
        "Modeling Baseline stage",
        "Advanced Modeling stage",
        "Baseline Inferance stage"
    ]
    parser.add_argument('--stages', nargs='+', required=True, choices=valid_stages,
                        help='List of stages to execute (space-separated)')
    args = parser.parse_args()
    stages_to_run = args.stages

    # Initialize configuration and MLflow tracking
    config = ConfigurationManager()
    mlflow_config = config.get_mlflow_config() 
    main_mlflow_tracker = MLflowTracker(config=mlflow_config)
    main_mlflow_tracker.start_run("MainPipeline")

    try:
        # Data Ingestion Stage
        if "Data Ingestion stage" in stages_to_run:
            STAGE_NAME = "Data Ingestion stage"
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                data_ingestion = DataIngestionPipeline(mlflow_tracker=main_mlflow_tracker)
                data_ingestion.main()
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

        # Data Profiling Stage
        if "Data Profiling stage" in stages_to_run:
            STAGE_NAME = "Data Profiling stage"
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                data_profiler = DataProfilerPipeline(mlflow_tracker=main_mlflow_tracker)
                data_profiler.main()
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

        # Data Transformation Stage
        if "Data Transformation stage" in stages_to_run:
            STAGE_NAME = "Data Transformation stage"
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                data_transformer = DataTransformerPipeline(mlflow_tracker=main_mlflow_tracker)
                data_transformer.main()
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

        # Baseline Modeling Stage
        if "Modeling Baseline stage" in stages_to_run:
            STAGE_NAME = "Modeling Baseline stage"
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                baseline_modeling = BaseLineModelingPipeline(mlflow_tracker=main_mlflow_tracker)
                baseline_modeling.main()
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

        # Advanced Modeling Stage
        if "Advanced Modeling stage" in stages_to_run:
            STAGE_NAME = "Advanced Modeling stage"
            X_test = ["This is a new text to predict that is good", "Another example text that is bad"]
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                transformer_model = TransformerModelPipeline(mlflow_tracker=main_mlflow_tracker)
                transformer_model.main(data=X_test)
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

        # Baseline Inference Stage
        if "Baseline Inferance stage" in stages_to_run:
            STAGE_NAME = "Baseline Inferance stage"
            model_name = "logistic_regression"
            stage = "Production"
            X_test = ["This is a new text to predict that is good", "Another example text that is bad"]
            try:
                logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<") 
                baseline_inference = BaseLineInferancePipeline(mlflow_tracker=main_mlflow_tracker)
                baseline_inference.main(model_name=model_name, stage=stage, data=X_test)
                logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
            except Exception as e:
                logger.exception(e)
                raise

    except Exception as e:
        logger.error("Pipeline execution failed with error: %s", str(e))
        raise
    finally:
        # Ensure MLflow run is ended even if there's an error
        main_mlflow_tracker.end_run()

if __name__ == '__main__':
    main()