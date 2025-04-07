from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..core import Cluster, ClusterName, ClustersConfig, QuestionId


class TagType(Enum):
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    HEADER = "HEADER"


@dataclass
class Tag:
    question_ids: list[QuestionId]
    tag_type: TagType
    match_string: str


Span = tuple[Optional[int], Optional[int]]


@dataclass
class TagsConfig:
    primary_regex: str = (
        r"^(Answer\s*(to))?\s*Question\s*(.+?)\s*(Ite(ra|ar)tion.+?)?\.\."
    )
    question_group: int = 3
    answer_to_group: int = 1
    question_id_regex: str = r"[0-9]+\s*\w?\s*[0-9]*"
    headers: list[str] = field(default_factory=list)


@dataclass
class Transcript:
    clusters: dict[ClusterName, Optional[Cluster]]

    def restore_cluster_data(self, clusters_cfg: ClustersConfig):
        for name, data in clusters_cfg.clusters.items():
            cluster = self.clusters.get(name, None)
            if cluster is not None:
                cluster.restore_data(data)
