from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_pipeline.data_profiler import SentimentDataProfiler
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger


class DataProfilerPipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker =mlflow_tracker
    def main(self):
        config = ConfigurationManager()
        data_profiler_config = config.get_data_profiler_config()
        profiler = SentimentDataProfiler(config=data_profiler_config, mlflow_tracker= self.mlflow_tracker)
        report_path = profiler.generate_report()
        logger.info(f"Data profiling report generated at {report_path}")

