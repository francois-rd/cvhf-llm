from dataclasses import dataclass, field, fields
from typing import Optional

import pandas as pd

from .parse_labels import ParsedLabelData
from .comparators import (
    Comparator,
    IntervalOrDateComparator,
    EnumComparator,
    ListOfStringsComparator,
    ListOfEnumsComparator,
    ListOrEnumComparator,
    MultiTagComparator,
    ScoreOrRejectComparator,
    Comparison,
    EnumComparison,
    StringComparison,
    ListOfStringsComparison,
    ListOfEnumsComparison,
    MultiTagComparison,
    ScoreComparison,
    ComparisonSate,
)

from ..core import ClusterName, ClustersConfig
from ..extract import ClusterOutput
from ..io import enum_from_str
from ..llms import Nickname
from ..parsing import ParserType


def comparator_factory(parser_type: str, tol: float = 0.0001) -> Comparator:
    parser_type = enum_from_str(ParserType, parser_type)
    if parser_type == ParserType.INTERVAL_OR_DATE_OR_REJECT:
        return IntervalOrDateComparator(tol=tol)
    elif parser_type == ParserType.ENUM:
        return EnumComparator()
    elif parser_type == ParserType.LIST_OF_STRINGS:
        return ListOfStringsComparator()
    elif parser_type == ParserType.LIST_OF_ENUMS:
        return ListOfEnumsComparator()
    elif parser_type == ParserType.LIST_OR_ENUM:
        return ListOrEnumComparator()
    elif parser_type == ParserType.MULTI_TAG:
        return MultiTagComparator()
    elif parser_type == ParserType.SCORE_OR_REJECT:
        return ScoreOrRejectComparator(tol=tol)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")


@dataclass
class RunningTally:
    total: float = 0
    count: int = 0

    def add(self, increment: float):
        self.total += increment
        self.count += 1


@dataclass
class ResultAtK:
    enum_recall: Optional[RunningTally] = None
    enum_macro_recall: Optional[RunningTally] = None
    string_recall: Optional[RunningTally] = None
    string_agreement: Optional[RunningTally] = None
    string_macro_recall: Optional[RunningTally] = None
    string_macro_agreement: Optional[RunningTally] = None
    tag_key_macro_recall: Optional[RunningTally] = None
    tag_value_macro_recall: Optional[RunningTally] = None
    tag_value_macro_agreement: Optional[RunningTally] = None
    score_recall: Optional[RunningTally] = None
    score_macro_difference: Optional[RunningTally] = None

    def add(self, field_name: str, increment: float):
        tally = getattr(self, field_name)
        if tally is None:
            tally = RunningTally()
            setattr(self, field_name, tally)
        tally.add(increment)

    def crunch(self) -> dict[str, float]:
        data = {}
        for f in fields(self):
            tally = getattr(self, f.name)
            data[f.name] = None if tally is None else tally.total / tally.count
        return data


@dataclass
class TopKComparison:
    few_shot: dict[ClusterName, list[Comparison]]
    other: dict[ClusterName, list[Comparison]]


@dataclass
class TopKResults:
    few_shot: list[pd.DataFrame]
    other: list[pd.DataFrame]


class GroundTruthComparator:
    def __init__(self, clusters: ClustersConfig, tol: float = 0.0001, k: int = 3):
        self.k = k
        self.comparators = {}
        self.few_shot_samples = {}
        for name, data in clusters.clusters.items():
            self.comparators[name] = comparator_factory(data.parser_type, tol=tol)
            self.few_shot_samples[name] = data.few_shot_assign_ids

    def __call__(
        self,
        assign_id: str,
        extraction_data: list[ClusterOutput],
        ground_truth: list[ParsedLabelData],
    ) -> TopKComparison:
        few_shot_results, other_results = {}, {}
        ground_truth = {data.name: data for data in ground_truth}
        for llm_output in extraction_data:
            truth = ground_truth.get(llm_output.cluster_name, None)
            is_few_shot = assign_id in self.few_shot_samples[llm_output.cluster_name]
            results = few_shot_results if is_few_shot else other_results
            results[llm_output.cluster_name] = self._compare(llm_output, truth)
        return TopKComparison(few_shot_results, other_results)

    def _compare(
        self,
        llm: ClusterOutput,
        truth: Optional[ParsedLabelData],
    ) -> list[Comparison]:
        results = []
        if truth is None:
            # This means we don't have any labels to compare against, so the
            # deepest k we can do is k=0.
            return results
        if llm.llm_output is None:
            # This means we don't need a detailed comparison: every k fails.
            k = min(self.k, len(truth.labels))
            return [Comparison.from_invalid_state(ComparisonSate.NO_LLM)] * k

        # General case: Make comparison for each k, then collate.
        for k in range(self.k):
            try:
                label = truth.labels[k]
            except IndexError:
                break
            results.append(self.comparators[llm.cluster_name](llm.llm_output, label))
        return results


@dataclass
class ValidationConfig:
    cluster_column_name: str = "cluster"
    run_ids_to_include: list[str] = field(default_factory=list)
    llms_to_include: list[Nickname] = field(default_factory=list)


class ComparisonAggregator:
    def __init__(self, cluster_column_name: str):
        self.cluster_column_name = cluster_column_name

    def __call__(self, comparisons_per_assign_id: list[TopKComparison]) -> TopKResults:
        few_shot_comps = [comp.few_shot for comp in comparisons_per_assign_id]
        other_comps = [comp.other for comp in comparisons_per_assign_id]
        return TopKResults(self._process(few_shot_comps), self._process(other_comps))

    def _process(
        self,
        comparisons_per_assign_id: list[dict[ClusterName, list[Comparison]]],
    ) -> list[pd.DataFrame]:
        max_k = 0
        top_k_per_assign_id_by_cluster: dict[ClusterName, list[list[Comparison]]] = {}
        for comparison in comparisons_per_assign_id:
            for name, top_k_for_one_assign_id in comparison.items():
                all_assign_id = top_k_per_assign_id_by_cluster.setdefault(name, [])
                all_assign_id.append(top_k_for_one_assign_id)
                max_k = max(max_k, len(top_k_for_one_assign_id))
        if max_k < 1:
            # No data to work with.
            return []

        all_data_per_top_k = [{} for _ in range(max_k)]
        for name, top_k_per_assign_id in top_k_per_assign_id_by_cluster.items():
            row_data_per_top_k = self._process_cluster(max_k, top_k_per_assign_id)
            self._fill_rows(name, all_data_per_top_k, row_data_per_top_k)
        for k in range(1, max_k):
            self._propagate_results(all_data_per_top_k[k - 1], all_data_per_top_k[k])
        dfs = []
        for data in all_data_per_top_k:
            dfs.append(pd.DataFrame(data).sort_values(by=self.cluster_column_name))
        return dfs

    def _fill_rows(
        self,
        name: ClusterName,
        all_data_per_top_k: list[dict[str, list[Optional[float]]]],
        row_data_per_top_k: list[ResultAtK],
    ) -> None:
        for all_data, row_data in zip(all_data_per_top_k, row_data_per_top_k):
            all_data.setdefault(self.cluster_column_name, []).append(name)
            for key, value in row_data.crunch().items():
                all_data.setdefault(key, []).append(value)

    def _process_cluster(
        self,
        max_k: int,
        top_k_per_assign_id: list[list[Comparison]],
    ) -> list[ResultAtK]:
        results_per_top_k = [ResultAtK() for _ in range(max_k)]
        for top_k_for_one_assign_id in top_k_per_assign_id:
            # NOTE: Repetition operator handles this gracefully.
            top_k_for_one_assign_id += [None] * (max_k - len(top_k_for_one_assign_id))

            self._tally_results(results_per_top_k, top_k_for_one_assign_id)
        return results_per_top_k

    def _tally_results(
        self,
        all_results: list[ResultAtK],
        top_k_for_one_assign_id: list[Optional[Comparison]],
    ) -> None:
        for result, comparison in zip(all_results, top_k_for_one_assign_id):
            for field_name, increment in self._process_at_k(comparison).items():
                result.add(field_name, increment)

    @staticmethod
    def _process_at_k(comparison: Optional[Comparison]) -> dict[str, float]:
        if comparison is None or not comparison.is_valid():
            return {}
        if isinstance(comparison, EnumComparison):
            return {"enum_recall": comparison.match}
        elif isinstance(comparison, StringComparison):
            return {
                "string_recall": comparison.exact,
                "string_agreement": comparison.agreement,
            }
        elif isinstance(comparison, ScoreComparison):
            return {
                "score_recall": comparison.exact,
                "score_macro_difference": comparison.difference,
            }
        elif isinstance(comparison, ListOfStringsComparison):
            return {
                "string_macro_recall": comparison.recall_exact,
                "string_macro_agreement": comparison.mean_agreement,
            }
        elif isinstance(comparison, ListOfEnumsComparison):
            return {"enum_macro_recall": comparison.recall_match}
        elif isinstance(comparison, MultiTagComparison):
            data = {"tag_key_macro_recall": comparison.recall_match_tag}
            # TODO: Do these count as tally(total+=0, count+=1)?
            #  They currently don't. To make them count, add:
            #     if <field> is None: data[<key>] = 0.0
            if comparison.recall_exact_value is not None:
                data["tag_value_macro_recall"] = comparison.recall_exact_value
            if comparison.mean_value_agreement is not None:
                data["tag_value_macro_agreement"] = comparison.mean_value_agreement
            return data
        elif type(comparison) is Comparison:
            raise ValueError(f"Base Comparison should never be valid: {comparison}")
        else:
            raise ValueError(f"Unsupported Comparison type: {comparison}")

    def _propagate_results(
        self,
        prev_data: dict[str, list[Optional[float]]],
        data: dict[str, list[Optional[float]]],
    ) -> None:
        # TODO: In theory, if some k1 < k2 has better scores, then k2 should be
        #  made to be a copy of k1. But that's for specific fields. In practice,
        #  there are several fields whose values are linked/grouped into specific
        #  sub-groups of fields. What happens when those sub-groups internally
        #  disagree on who is better?
        #  Current solution: Treat all fields as independent (even if not quite right).

        #  NOTE: The reason we only need the immediate previous data is that the
        #  propagation is monotonic: if field_at(k-1).is_better_than(field_at(k)),
        #  then field_at(k) = field_at(k-1). So then k+1 only needs to look at
        #  k to fix itself (since any improvement from k-1 already propagated).
        for prev_i, cluster_name in enumerate(prev_data[self.cluster_column_name]):
            data_i = data[self.cluster_column_name].index(cluster_name)
            self._propagate_row(prev_i, data_i, prev_data, data)

    def _propagate_row(
        self,
        prev_i: int,
        data_i: int,
        prev_data: dict[str, list[Optional[float]]],
        data: dict[str, list[Optional[float]]],
    ) -> None:
        for column_name in prev_data:
            if column_name == self.cluster_column_name:
                continue
            prev_val = prev_data[column_name][prev_i]
            data_val = data[column_name][data_i]
            less = column_name == "score_macro_difference"
            if self._do_overwrite_data(prev_val, data_val, less):
                data[column_name][data_i] = prev_data[column_name][prev_i]

    @staticmethod
    def _do_overwrite_data(
        prev_val: Optional[float],
        data_val: Optional[float],
        less: bool,
    ) -> bool:
        if prev_val is None:
            return False
        if data_val is None:
            return True
        prev_is_better = prev_val < data_val if less else prev_val > data_val
        return prev_is_better
