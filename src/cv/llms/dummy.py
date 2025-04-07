from dataclasses import dataclass

from .base import LLM, LLMOutput, Nickname
from ..prompting import ClusterPrompt


@dataclass
class DummyConfig:
    dummy_output: str = ""
    cheat: bool = False


class DummyLLM(LLM):
    def __init__(self, nickname: Nickname, llm_cfg: DummyConfig, *args, **kwargs):
        super().__init__(nickname, llm_cfg, *args, **kwargs)
        self.cfg = llm_cfg
        self.sampler = kwargs["dummy_cheat_sampler"]

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        if self.cfg.cheat:
            label = self.sampler.get(prompt.name, kwargs["assign_id"]).label
            return LLMOutput(generated_text=label or "", error_message=None)
        return LLMOutput(generated_text=self.cfg.dummy_output, error_message=None)
