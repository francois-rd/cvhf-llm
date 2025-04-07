from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ..prompting import ClusterPrompt


# A name to uniquely differentiate different LLMs.
Nickname = str
MISSING_NICKNAME: Nickname = "<missing-llm>"


class LLMImplementation(Enum):
    MISSING = "MISSING"
    DUMMY = "DUMMY"
    HF_TRANSFORMERS = "HF_TRANSFORMERS"


@dataclass
class LLMOutput:
    # The LLM's generated output text, or None if an error occurred.
    generated_text: Optional[str]

    # Non-None only if an error occurred, in which case the error message is given here.
    error_message: Optional[str]


class LLM:
    """Wrapper interface for all LLM implementations."""

    def __init__(self, nickname: Nickname, *args, **kwargs):
        self.nickname = nickname

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        """Invokes the underlying LLM, returning its output."""
        raise NotImplementedError


@dataclass
class LLMsConfig:
    # The nickname of the LLM to load.
    llm: Nickname = MISSING_NICKNAME

    # The implementation to use to load the LLM.
    implementation: LLMImplementation = LLMImplementation.MISSING
