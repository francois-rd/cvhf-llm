from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .base import LLM, LLMOutput, Nickname
from ..prompting import ClusterPrompt, Message, MessageType


@dataclass
class TransformersConfig:
    # RNG seed for replication of results.
    seed: int = 314159

    # Whether to split out the system instructions (for LLMs that support it).
    use_system_prompt: bool = False

    # Whether to use the transformers.Pipeline chat templating functionality.
    use_chat: bool = False

    # Conversion between each MessageType and the specific role name for this LLM.
    message_type_to_role_map: dict[MessageType, str] = field(
        default_factory=lambda: {
            MessageType.SYSTEM: MessageType.SYSTEM.value.lower(),
            MessageType.USER: MessageType.USER.value.lower(),
            MessageType.ASSISTANT: MessageType.ASSISTANT.value.lower(),
        }
    )

    # Suffix for each MessageType's template when joining them all into one string.
    message_type_suffix: dict[MessageType, str] = field(
        default_factory=lambda: {
            MessageType.SYSTEM: "\n\n",
            MessageType.USER: "\n",
            MessageType.ASSISTANT: "\n\n",
        }
    )

    # Model quantization options for bitsandbytes.
    quantization: Optional[str] = None

    # See transformers.AutoModelForCausalLM.from_pretrained for details.
    # NOTE: Skip 'quantization_config', which is handled specially.
    model_params: dict[str, Any] = field(
        default_factory=lambda: {"trust_remote_code": True},
    )

    # See transformers.Pipeline for details.
    # NOTE: Skip 'task', 'model', and 'torch_dtype', which are handled specially.
    pipeline_params: dict[str, Any] = field(default_factory=dict)

    # See transformers.GenerationConfig for details.
    generation_params: dict[str, Any] = field(
        default_factory=lambda: {
            "return_full_text": False,
            "max_new_tokens": 50,
            "num_return_sequences": 1,
        },
    )


class TransformersLLM(LLM):
    """Interface for all LLMs using the HuggingFace transformers.Pipeline API."""

    def __init__(
        self,
        nickname: Nickname,
        llm_cfg: TransformersConfig,
        *args,
        **kwargs,
    ):
        # Delayed imports.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
            set_seed,
        )

        # Basic initialization.
        set_seed(llm_cfg.seed)
        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg

        # Quantization.
        model_params = {**self.cfg.model_params}  # Convert OmegaConf -> dict
        if self.cfg.quantization is not None:
            if self.cfg.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.cfg.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unsupported quantization: {self.cfg.quantization}")
            model_params.update({"quantization_config": bnb_config})

        # Pipeline initialization.
        self.llm = pipeline(
            task="text-generation",
            model=AutoModelForCausalLM.from_pretrained(nickname, **model_params),
            tokenizer=AutoTokenizer.from_pretrained(nickname),
            torch_dtype=torch.bfloat16,
            **self.cfg.pipeline_params,
        )

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        if prompt.messages is None:
            return LLMOutput(None, "Missing prompt messages.")
        prompt = self._make_prompt(prompt.messages)
        output = self.llm(prompt, **self.cfg.generation_params)
        generated_text = output[0]["generated_text"]
        return LLMOutput(generated_text, error_message=None)

    def _make_prompt(self, messages: list[Message]) -> Union[str, list[dict[str, str]]]:
        roles, suffix = self.cfg.message_type_to_role_map, self.cfg.message_type_suffix
        new_messages = [{"role": roles[m[0]], "content": m[1]} for m in messages]
        text = "".join([f"{m[1]}{suffix[m[0]]}" for m in messages]).strip()
        if self.cfg.use_chat:
            if self.cfg.use_system_prompt:
                prompt = new_messages
            else:
                prompt = [{"role": "user", "content": text}]
        else:
            prompt = text
        return prompt
