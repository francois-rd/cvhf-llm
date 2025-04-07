from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf


ClusterName = str
Lines = list[str]
Prompt = str
QuestionId = str


@dataclass
class ClusterData:
    prompt: Prompt
    question_ids: list[QuestionId]
    parser_type: str
    few_shot_assign_ids: list[str] = field(default_factory=list)
    system_prompt_index: int = 0
    parser_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClustersConfig:
    system_prompt_options: list[Prompt] = field(default_factory=list)
    instruction_prompt_to_sample_delimiter: str = "\n\n"
    sample_template: Prompt = "*Snippet of transcript*\n{transcript}\n\n*Answer*"
    included_clusters: list[ClusterName] = field(default_factory=list)
    clusters: dict[ClusterName, ClusterData] = field(default_factory=dict)


@dataclass
class Cluster:
    data: Optional[ClusterData] = None
    lines: Lines = field(default_factory=list)

    def restore_data(self, data: ClusterData):
        self.data = OmegaConf.to_object(data)
