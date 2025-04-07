from typing import Any, Callable, Optional
from dataclasses import dataclass

from ..core import ClusterName, ClustersConfig, Label, LabelData
from ..parsing import ParserManager


@dataclass
class ParsedLabelData:
    name: ClusterName
    labels: list[Any]


class GroundTruthParser:
    def __init__(self, clusters: ClustersConfig):
        self.parsers = ParserManager(clusters)

    def __call__(
        self,
        assign_id: str,
        label_data: LabelData,
        *args,
        **kwargs,
    ) -> list[ParsedLabelData]:
        result = {}
        for name, data in label_data.items():
            if data is None:
                continue
            parser_output = self._parse_all(
                # Data is expected to be of type: Label or list[Label].
                data=[data] if isinstance(data, Label) else data,
                parser=lambda label: self.parsers.get(name)(label, *args, **kwargs),
                error_msg=f"{assign_id}: {name}: Parsing failed: " + "{}",
            )
            result.setdefault(name, []).extend(parser_output)
        return [ParsedLabelData(name=n, labels=r) for n, r in result.items()]

    @staticmethod
    def _parse_all(
        data: list[Label],
        parser: Callable[[Label], Optional[Any]],
        error_msg: str,
    ) -> list[Any]:
        output = []
        for label in data:
            parser_output = parser(label)
            if parser_output is None:
                raise ValueError(error_msg.format(label))
            output.append(parser_output)
        return output
