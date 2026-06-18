"""
OpenTelemetry instrumentation for Mental Health RAG API.

3 Metrics:
  1. NLP    — nlp.intent_requests       : counter per intent + emotion label
  2. Data   — data.message_length       : histogram of user message char length
  3. Server — server.http_requests_total: counter per endpoint + status code
"""

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
import os

resource = Resource.create({"service.name": "nlp-backend"})

exporter = OTLPMetricExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
    #endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True,
)

reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15_000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter("nlp-backend")

# 1. NLP metric — intent distribution (also tagged with emotion for richer analysis)
intent_counter = meter.create_counter(
    name="nlp.intent_requests",
    description="Number of requests per predicted intent and emotion label",
    unit="1",
)

# 2. Data metric — message length distribution
message_length_histogram = meter.create_histogram(
    name="data.message_length",
    description="Distribution of user message lengths in characters",
    unit="chars",
)

# 3. Server metric — request count by endpoint and status code
request_counter = meter.create_counter(
    name="server.http_requests_total",
    description="Total HTTP requests by endpoint and status code",
    unit="1",
)


def record_intent(intent: str, emotion: str = "unknown", language: str = "unknown"):
    intent_counter.add(1, {"intent": intent, "emotion": emotion, "language": language})


def record_message_length(text: str):
    message_length_histogram.record(len(text))


def record_request(endpoint: str, status_code: int):
    request_counter.add(1, {
        "endpoint": endpoint,
        "status_code": str(status_code),
        "status_class": f"{status_code // 100}xx",
    })
