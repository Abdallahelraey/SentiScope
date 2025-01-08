from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_profiler import SentimentDataProfiler
from SentiScope.logging import logger


class DataProfilerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_profiler_config = config.get_data_profiler_config()
        profiler = SentimentDataProfiler(config=data_profiler_config)
        report_path = profiler.generate_report()
        logger.info(f"Data profiling report generated at {report_path}")

