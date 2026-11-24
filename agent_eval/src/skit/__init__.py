from dotenv import load_dotenv
from skit.pipelines.pdf_qa_gen import PDFGroundedQAGeneratedResponse

load_dotenv()
import logfire
from logfire import ConsoleOptions
import os

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
logfire.configure(
    scrubbing=False,
    service_name="skit",
    # console=ConsoleOptions(
    #     span_style="simple",
    # ),
)
logfire.instrument_openai_agents()
logfire.instrument_openai()
logfire.instrument_litellm()
logfire.instrument_google_genai()


# logfire.info("Loading .env file")


__all__ = ["PDFGroundedQAGeneratedResponse"]

