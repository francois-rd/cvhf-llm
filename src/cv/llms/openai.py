# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Optional

from retry.api import retry_call

from .base import LLM, LLMOutput, Nickname
from ..prompting import ClusterPrompt, Message, MessageType


@dataclass
class OpenAIConfig:
    # The base URL of the local vLLM server hosting the local LLM.
    vllm_base_url: str = "<missing>"

    # Whether to split out the system instructions (for LLMs that support it).
    use_system_prompt: bool = False

    # Whether to use the OpenAI Completions or OpenAI ChatCompletions protocol.
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

    # See openai.OpenAI.chat.completions.create for details. Skip 'model' and
    # 'messages' which are handled specially. Other options may exist when servicing
    # requests based on vLLM servers. See:
    # - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    chat_completion_query_params: dict[str, Any] = field(
        default_factory=lambda: {
            # New tokens to add, not total tokens.
            "max_completion_tokens": 50,
            # Experimental feature in OpenAI. Might not be 100% deterministic.
            "seed": 314159,
        },
    )

    # See openai.OpenAI.completions.create for details. Skip 'model' and 'prompt' which
    # are handled specially. Other options may exist when servicing requests based on
    # vLLM servers. See: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    completion_query_params: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 50,
            # Experimental feature in OpenAI.
            "seed": 314159,
        },
    )


@dataclass
class _InvocationOutput:
    generated_text: Optional[str]
    refusal: Optional[bool]


class OpenAILLM(LLM):
    """
    Interface for all LLMs using the OpenAI API to send requests to a vLLM server.
    """

    def __init__(
        self,
        nickname: Nickname,
        llm_cfg: OpenAIConfig,
        *args,
        **kwargs,
    ):
        from openai import OpenAI  # Delayed import.

        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg
        self.client = OpenAI(base_url=self.cfg.vllm_base_url, api_key="EMPTY")

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        from openai import OpenAIError

        if prompt.messages is None:
            return LLMOutput(None, "Missing prompt messages.")

        try:
            f_args = (prompt.messages,)
            out = retry_call(self._do_invoke, fargs=f_args, fkwargs=kwargs, delay=10)
        except OpenAIError as e:
            return LLMOutput(generated_text=None, error_message=str(e))
        if out.refusal:
            return LLMOutput(generated_text=None, error_message=out.generated_text)
        return LLMOutput(generated_text=out.generated_text, error_message=None)

    def _do_invoke(self, messages: list[Message], **_) -> _InvocationOutput:
        roles, suffix = self.cfg.message_type_to_role_map, self.cfg.message_type_suffix
        new_messages = [{"role": roles[m[0]], "content": m[1]} for m in messages]
        text = "".join([f"{m[1]}{suffix[m[0]]}" for m in messages]).strip()
        if self.cfg.use_chat:
            if self.cfg.use_system_prompt:
                messages = new_messages
            else:
                messages = [{"role": "user", "content": text}]
            response = self.client.chat.completions.create(
                model=self.nickname,
                messages=messages,
                **self.cfg.chat_completion_query_params,
            )

            # This is always a string, or None if refusal is not None.
            generated_text = response.choices[0].message.content

            # This is presumably non-None when a refusal happens.
            refusal = response.choices[0].message.refusal is not None
        else:
            response = self.client.completions.create(
                model=self.nickname,
                prompt=text,
                **self.cfg.completion_query_params,
            )

            # This is always a string. If content filtering removed it, then it might
            # be an empty string. Who knows. It is not specified in the documentation.
            generated_text = response.choices[0].text

            # This is presumably the only time a refusal happens.
            refusal = response.choices[0].finish_reason == "content_filter"
        return _InvocationOutput(generated_text, refusal)
