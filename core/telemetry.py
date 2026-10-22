"""Optional OTLP tracing with allowlisted non-content attributes."""
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace

from config.settings import settings

_provider = None


def initialize():
    global _provider
    if not settings.OTEL_ENABLED or _provider is not None:
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _provider = TracerProvider(resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME}))
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)))
    trace.set_tracer_provider(_provider)


def shutdown():
    if _provider:
        _provider.shutdown()


@contextmanager
def span(name, context=None):
    with trace.get_tracer("rag").start_as_current_span(name, context=context, record_exception=False, set_status_on_exception=False) as current:
        try:
            yield current
        except Exception as exc:
            current.set_attribute("error.type", type(exc).__name__)
            current.set_status(trace.Status(trace.StatusCode.ERROR))
            raise


def traced(name):
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            with span(name):
                return function(*args, **kwargs)
        return invoke
    return decorate
