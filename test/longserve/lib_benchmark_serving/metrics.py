import math
import numpy as np
import dataclasses

from lib_benchmark_serving.structs import TestRequest, Dataset, ReqResult

@dataclasses.dataclass
class BenchmarkMetrics:
    num_requests: int
    test_duration_s: float

    request_throughput: float
    avg_inputlen: float
    avg_outputlen: float
    avg_per_token_latency_ms: float
    avg_input_token_latency_ms: float
    avg_output_token_latency_ms: float
    
    avg_latency: float
    TTFT: float
    TTFT_median: float
    TTFT_p99: float

    TPOT: float
    TPOT_median: float
    TPOT_p99: float

    @staticmethod
    def from_req_results(data: list[ReqResult]):
        test_duration_ms = float(np.max([req.complete_time for req in data]) - np.min([req.issue_time for req in data]))*1000
        return BenchmarkMetrics(
            len(data),
            test_duration_ms / 1000,
            len(data) / (test_duration_ms / 1000),
            float(np.mean([(req.prompt_len) for req in data])),
            float(np.mean([(req.output_len) for req in data])),
            float(np.mean([req.latency / (req.prompt_len+req.output_len) for req in data]))*1000,
            float(np.mean([req.ttft / req.prompt_len for req in data]))*1000,
            float(np.mean([req.tpot for req in data if req.output_len != 0]))*1000,
            float(np.mean([req.latency for req in data]))*1000,

            float(np.mean([req.ttft for req in data]))*1000,
            float(np.median([req.ttft for req in data]))*1000,
            float(np.percentile([req.ttft for req in data],99))*1000,

            float(np.mean([req.tpot for req in data if req.output_len != 0]))*1000,
            float(np.median([req.tpot for req in data if req.output_len != 0]))*1000,
            float(np.percentile([req.tpot for req in data if req.output_len != 0],99))*1000
        )
    
    def __str__(self) -> str:
        result = "{\n"
        for field in dataclasses.fields(self):
            result += f"\t{field.name}: {getattr(self, field.name)}\n"
        result += "}"
        return result
