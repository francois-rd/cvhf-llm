from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ..core import Cluster, ClusterName, ClustersConfig, FewShotSampler, Lines
from ..io import scrub
from ..segmentation import Transcript


class MessageType(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"


Message = tuple[MessageType, str]


@dataclass
class ClusterPrompt:
    # The name of the cluster.
    name: ClusterName

    # The langchain-like list of template messages for the cluster.
    messages: Optional[list[Message]]


class PromptMaker:
    def __init__(self, cfg: ClustersConfig, sampler: FewShotSampler):
        self.cfg = cfg
        self.sampler = sampler

    def __call__(self, transcript: Transcript, *args, **kwargs) -> list[ClusterPrompt]:
        results = []
        for name, cluster in self._get_included_clusters(transcript).items():
            if cluster is None:
                results.append(ClusterPrompt(name, None))
            else:
                template = self._make_template(name, cluster, **kwargs)
                results.append(ClusterPrompt(name, template))
        return results

    def _get_included_clusters(
        self,
        transcript: Transcript,
    ) -> dict[ClusterName, Cluster]:
        included = self.cfg.included_clusters
        return {k: v for k, v in transcript.clusters.items() if k in included}

    def _make_template(
        self,
        name: ClusterName,
        cluster: Cluster,
        **kwargs,
    ) -> list[Message]:
        system_prompt = self.cfg.system_prompt_options[cluster.data.system_prompt_index]
        messages = [(MessageType.SYSTEM, system_prompt.format(**kwargs))]
        instruction = cluster.data.prompt.format(**kwargs)
        delimiter = self.cfg.instruction_prompt_to_sample_delimiter
        first, a_ids = True, cluster.data.few_shot_assign_ids
        for a_id in a_ids:
            data = self.sampler.get(name, a_id)
            if data is None:
                raise ValueError(
                    f"Missing label {name} ({a_id}): Cannot generate few-shot sample."
                )
            template = self._make_sample(data.lines, first, instruction, delimiter)
            messages.append((MessageType.USER, template))
            messages.append((MessageType.ASSISTANT, str(data.label)))
            first = False
        # NOTE: Yes, we reuse first here deliberately.
        template = self._make_sample(cluster.lines, first, instruction, delimiter)
        messages.append((MessageType.USER, template))
        return messages

    def _make_sample(
        self,
        lines: Lines,
        is_first: bool,
        instruction: str,
        delimiter: str,
    ) -> str:
        template = self.cfg.sample_template.format(transcript=scrub("\n".join(lines)))
        if is_first:
            template = f"{instruction}{delimiter}{template}"
        return template
