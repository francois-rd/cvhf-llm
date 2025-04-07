import os.path
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..core import ClusterName, ClustersConfig
from ..extract import ClusterOutput, ExtractionDirHandler
from ..io import ensure_path, enum_from_str
from ..llms import Nickname
from ..parsing import (
    IntervalOrDateOrReject,
    ListOrEnum,
    ParserType,
    ScoreOrReject,
    Tag,
)


@dataclass
class ConsolidateConfig:
    assign_id_column_name: str = "Assign_ID"
    llm_column_name: str = "LLM"
    list_item_separator: str = "|"
    ordered_run_ids_old_to_new: list[str] = field(default_factory=list)
    llms_to_include: list[Nickname] = field(default_factory=list)
    assign_id_blacklist: list[str] = field(default_factory=list)


@dataclass
class _IntermediaryResult:
    run_id: str
    llm: str
    data: dict[ClusterName, ClusterOutput]


@dataclass
class _CleanResult:
    llm: str
    data: dict[ClusterName, Optional[Any]]


class Consolidator:
    def __init__(
        self,
        cfg: ConsolidateConfig,
        clusters_cfg: ClustersConfig,
        pathing: ExtractionDirHandler,
    ):
        self.cfg = cfg
        self.clusters = clusters_cfg
        self.pathing = pathing
        self.reversed_run_ids = list(reversed(cfg.ordered_run_ids_old_to_new))
        self.parser_types = {
            name: enum_from_str(ParserType, data.parser_type)
            for name, data in clusters_cfg.clusters.items()
        }

    def __call__(self, output_dir: str, *args, **kwargs):
        """
        Part of the consolidation is to ensure that TAGS and ENUMS are always in
        uppercase (assuming they are specified as such in ClustersConfig). All
        other data is either: a number (int/float), a date (ISO format), or a
        lowercase string. In particular, tag values are lowercased as 'TAG: value'.
        Missing data is either an empty string or float('nan') depending on the
        CSV reading method (e.g., pd.read_csv() uses float('nan') by default).
        Also, where the output is a list, we collapse it into a string and use
        ConsolidateConfig.list_item_separator to delineate the list items.

        Parsing these data back into can be somewhat complex. One simple algorithm
        is as follows:
        1. Check if the entry is missing (either emtpy string or float('nan')).
        2. Check if the entry is a number (either directly a float or a string that
           can be coerced to a float).
        3. Check if the entry can be coerced to an ISO date (YYYY-MM-DD) using datetime.
        4. Check if the entry contains 'ConsolidateConfig.list_item_separator'.
           If so, apply 5-7 below on EACH ITEM in the list generated from str.split().
        5. Check if the entry is all UPPERCASE. If so, it is a categorical variable.
        6. Check if the entry is all lowercase. If so, it is a plain list item.
        7. Otherwise, the entry should have a 'TAG: value' format. Split on ': '
           if you need to extrac the TAG or the value individually.

        NOTE: This algorithm is the simplest way to plainly extract the data. If
        you need to do any validation or want to process the data further (e.g.,
        interval vs date), then you need a specialized processor for each cluster
        type. That information is in ClustersConfig (each cluster name has a type).
        See cv/analyze/histogram.py for processing example. There, we need to track
        entries by type in order to have sensible/type-aware histogram generation.
        """
        result_by_aid = self._get_intermediary_results()
        result_by_aid = {k: self._keep_only_latest(v) for k, v in result_by_aid.items()}
        output_file = os.path.join(output_dir, f"{self.reversed_run_ids[0]}.csv")
        self._to_df(result_by_aid).to_csv(ensure_path(output_file), index=False)

    def _get_intermediary_results(
        self,
    ) -> dict[str, dict[str, dict[str, _IntermediaryResult]]]:
        result_by_aid = {}
        for data, walk in self.pathing.walk_extractions():
            a_id = self.pathing.get_assign_id(walk)
            if a_id in self.cfg.assign_id_blacklist:
                continue
            run_id, llm = self.pathing.get_run_id_and_llm(walk)
            if run_id not in self.cfg.ordered_run_ids_old_to_new:
                continue
            if llm not in self.cfg.llms_to_include:
                continue
            data = {c.cluster_name: c for c in data}
            result = _IntermediaryResult(run_id, llm, data)
            result_by_aid.setdefault(a_id, {}).setdefault(llm, {})[run_id] = result
        return result_by_aid

    def _keep_only_latest(
        self,
        aid_result_per_llm: dict[str, dict[str, _IntermediaryResult]],
    ) -> list[_CleanResult]:
        results = []
        for llm, aid_result in aid_result_per_llm.items():
            results.append(self._keep_latest_for_llm(llm, aid_result))
        return results

    def _keep_latest_for_llm(
        self,
        llm: str,
        aid_result: dict[str, _IntermediaryResult],
    ) -> _CleanResult:
        latest = {}
        for name in self.clusters.included_clusters:
            if name not in latest:
                latest[name] = self._find_latest(name, aid_result)
        return _CleanResult(llm, latest)

    def _find_latest(
        self,
        name: ClusterName,
        aid_result: dict[str, _IntermediaryResult],
    ) -> Optional[Any]:
        for run_id in self.reversed_run_ids:
            if run_id not in aid_result:
                continue
            intermediary = aid_result[run_id]
            if name in intermediary.data:
                cluster_output = intermediary.data[name]
                # Return immediately to ensure that we keep data
                # from only the latest run ID that matches.
                return cluster_output.llm_output
        return None  # If data is entirely missing, mark is as not found.

    def _to_df(self, clean_results: dict[str, list[_CleanResult]]) -> pd.DataFrame:
        data = {}
        for a_id, result_list in clean_results.items():
            self._add_data_row(a_id, data, result_list)
        cols = [self.cfg.assign_id_column_name, self.cfg.llm_column_name]
        return pd.DataFrame(data).sort_values(by=cols).sort_index(axis=1)

    def _add_data_row(
        self,
        assign_id: str,
        data: dict[str, list],
        clean_results: list[_CleanResult],
    ) -> None:
        for result in clean_results:
            data.setdefault(self.cfg.assign_id_column_name, []).append(assign_id)
            data.setdefault(self.cfg.llm_column_name, []).append(result.llm)
            for name, value in result.data.items():
                data.setdefault(name, []).append(self._to_primitive(name, value))

    def _to_primitive(self, name: ClusterName, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        error_message = "At least one field must be non-None, but got: {}"
        parser_type = self.parser_types[name]
        if parser_type == ParserType.INTERVAL_OR_DATE_OR_REJECT:
            value = IntervalOrDateOrReject(**value)
            if value.interval is not None:
                return str(value.interval)
            elif value.date is not None:
                return value.date
            elif value.reject is not None:
                return value.reject
            else:
                raise ValueError(error_message.format(value))
        elif parser_type == ParserType.ENUM:
            return value  # Already converted to string.
        elif parser_type == ParserType.LIST_OF_STRINGS:
            return self.cfg.list_item_separator.join([v.lower() for v in value])
        elif parser_type == ParserType.LIST_OF_ENUMS:
            # Enums are already str in this reloaded format, so lower() works.
            return self.cfg.list_item_separator.join([v.lower() for v in value])
        elif parser_type == ParserType.LIST_OR_ENUM:
            value = ListOrEnum(**value)
            if value.strings is not None:
                value = [v.lower() for v in value.strings]
                return self.cfg.list_item_separator.join(value)
            elif value.enum_value is not None:
                return value.enum_value
            else:
                raise ValueError(error_message.format(value))
        elif parser_type == ParserType.MULTI_TAG:
            value = [Tag(**data).to_string() for data in value]
            return self.cfg.list_item_separator.join(value)
        elif parser_type == ParserType.SCORE_OR_REJECT:
            value = ScoreOrReject(**value)
            if value.score is not None:
                return str(value.score)
            elif value.reject is not None:
                return value.reject
            else:
                raise ValueError(error_message.format(value))
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")
