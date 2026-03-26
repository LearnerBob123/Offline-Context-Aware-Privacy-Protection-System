"""
Abstract base class for all pipeline modules.
Enforces a consistent interface across detectors, embedders, engines, etc.
"""

from abc import ABC, abstractmethod
from config.config import PipelineConfig


class BaseModule(ABC):
    """Every pipeline component inherits from this."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Load models / allocate resources."""

    @abstractmethod
    def release(self) -> None:
        """Free GPU memory / close handles."""
