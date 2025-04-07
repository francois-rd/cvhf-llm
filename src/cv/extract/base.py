from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.runnables import Runnable, chain
from langchain_core.runnables.base import RunnableEach

from ..core import ClusterName, ClustersConfig, FewShotSampler
from ..llms import LLMsConfig, load_llm
from ..parsing import ParserManager
from ..prompting import ClusterPrompt, PromptMaker
from ..segmentation import Transcript


@dataclass
class ClusterOutput:
    cluster_name: ClusterName
    llm_output: Optional[Any]
    error_message: Optional[str]


@dataclass
class TranscriptWrapper:
    """
    Surely, there is a better way to pass args and kwargs to each Runnable in a
    Chain, but I cannot figure it out from the documentation.
    """

    transcript: Transcript
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class PromptWrapper:
    """
    Surely, there is a better way to pass args and kwargs to each Runnable in a
    Chain, but I cannot figure it out from the documentation.
    """

    prompt: ClusterPrompt
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class Extract:
    def __init__(
        self,
        clusters: ClustersConfig,
        llms: LLMsConfig,
        sampler: FewShotSampler,
        *args,
        **kwargs,
    ):
        self.clusters_cfg = clusters
        self.make_prompts = PromptMaker(clusters, sampler)
        self.parsers = ParserManager(clusters)
        self.llm = load_llm(llms, *args, **kwargs)
        self.chain = self._generate_chain()

    def __call__(self, transcript: Transcript, *args, **kwargs) -> list[ClusterOutput]:
        transcript.restore_cluster_data(self.clusters_cfg)
        return self.chain.invoke(TranscriptWrapper(transcript, args, kwargs))

    def _generate_preprocessor(self) -> Runnable:
        @chain
        def runnable(w: TranscriptWrapper) -> list[PromptWrapper]:
            prompts = self.make_prompts(w.transcript, *w.args, **w.kwargs)
            return [PromptWrapper(prompt, w.args, w.kwargs) for prompt in prompts]

        return runnable

    def _generate_llm(self) -> Runnable:
        @chain
        def runnable(w: PromptWrapper) -> ClusterOutput:
            if w.prompt.messages is None:
                # If there is no cluster data for whatever reason (parsing error or
                # legitimately missing data from the transcript), skip the LLM.
                return ClusterOutput(w.prompt.name, None, "No cluster data.")
            output = self.llm.invoke(w.prompt, *w.args, **w.kwargs)
            parser = self.parsers.get(w.prompt.name)
            return ClusterOutput(
                cluster_name=w.prompt.name,
                llm_output=parser(output.generated_text, *w.args, **w.kwargs),
                error_message=output.error_message,
            )

        return runnable

    def _generate_chain(self) -> Runnable:
        return self._generate_preprocessor() | RunnableEach(bound=self._generate_llm())
