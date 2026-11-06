class DatasetBuilderError(Exception):
    """Base error for actionable project failures."""


class ConfigurationError(DatasetBuilderError):
    """Raised when configuration is invalid or incomplete."""


class SourceError(DatasetBuilderError):
    """Raised when a source cannot be safely imported or verified."""


class KnowledgeBaseError(DatasetBuilderError):
    """Raised when reference data cannot support a verified knowledge base."""


class PlanningError(DatasetBuilderError):
    """Raised when a generation plan or its durable state is invalid."""



class LLMConfigurationError(DatasetBuilderError):
    """Raised when Azure OpenAI settings are missing, unsafe, or rejected as permanent."""


class LLMRequestError(DatasetBuilderError):
    """Raised when a model request fails permanently or exhausts its bounded retries."""


class LLMRefusalError(LLMRequestError):
    """Raised when the model or content filter declines to produce content."""


class GenerationError(DatasetBuilderError):
    """Raised when generation inputs, budgets, or prerequisites are invalid."""


class ReviewError(DatasetBuilderError):
    """Raised when review, acceptance, or finalisation prerequisites are not met."""
