from typing import Optional, Union
from dataclasses import dataclass

from .cluster import Cluster, ClusterName, Lines
from ..io import PathConfig, load_dataclass_json, walk_json
from ..segmentation import Transcript


Label = str
LabelData = dict[ClusterName, Optional[Union[Label, list[Label]]]]


@dataclass
class SampleData:
    lines: Lines
    label: Optional[Label]


class FewShotSampler:
    def __init__(self, paths: PathConfig):
        self.data_by_assign_id = {}
        for label_data, walk in walk_json(paths.labeled_transcript_dir):
            a_id = walk.no_ext()
            file_path = walk.map(paths.clustered_transcript_dir)
            transcript = load_dataclass_json(file_path, t=Transcript)
            sample_data = self._make_sample_data(a_id, transcript, label_data)
            self.data_by_assign_id[a_id] = sample_data

    def _make_sample_data(
        self,
        assign_id: str,
        transcript: Transcript,
        label_data: LabelData,
    ) -> dict[ClusterName, Optional[SampleData]]:
        result = {}
        for name, c in transcript.clusters.items():
            result[name] = self._do_make_sample_data(assign_id, name, c, label_data)
        return result

    @staticmethod
    def _do_make_sample_data(
        assign_id: str,
        name: ClusterName,
        cluster: Optional[Cluster],
        label_data: LabelData,
    ) -> Optional[SampleData]:
        if cluster is None:
            return None
        try:
            label = label_data[name]
        except KeyError:
            raise ValueError(f"{assign_id} is missing '{name}' cluster label.")
        if label is None or isinstance(label, Label):
            return SampleData(cluster.lines, label)
        elif isinstance(label, list):
            return SampleData(cluster.lines, label[0])
        else:
            raise ValueError(
                f"{assign_id}: cluster '{name}': Label must be None, a plain "
                f"label, or a list of labels, not {type(label)}"
            )

    def get(self, name: ClusterName, assign_id: str) -> Optional[SampleData]:
        try:
            return self.data_by_assign_id[assign_id][name]
        except KeyError:
            raise ValueError(f"{assign_id} is missing '{name}' cluster label.")
