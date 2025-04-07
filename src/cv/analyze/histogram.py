from typing import Any, Iterable, Optional, Sized, Union
from dataclasses import dataclass, Field, field, fields
import math

import pandas as pd

from ..consolidate import ConsolidateConfig
from ..core import ClusterName, ClustersConfig
from ..io import enum_from_str
from ..llms import Nickname
from ..parsing import (
    EnumParser,
    IntervalOrDateParser,
    MultiTagParser,
    ParserType,
    ScoreOrRejectParser,
    Tag,
)


Datum = Union[float, str]
Data = list[Datum]
PlainHistogram = dict[Datum, int]


@dataclass
class Histogram:
    cluster_name: ClusterName
    llm_nickname: Nickname
    total_count: int
    originals: dict[Datum, Datum] = field(default_factory=dict)

    def make_report(self, indent: str = "    ") -> str:
        header = f"Cluster ({self.cluster_name}) + LLM ({self.llm_nickname})"
        header = self._report_header(header, indent=indent, indents=0)
        blacklist = [f.name for f in fields(Histogram)]
        span = self._length_span(
            [f.name for f in fields(self) if f.type is int and f.name not in blacklist],
        )
        field_reports = [
            self._report_field(f, span, indent=indent)
            for f in fields(self)
            if f.name not in blacklist
        ]
        return header + "".join(field_reports)

    def _report_field(
        self,
        f: Field,
        top_level_span: tuple[int, int],
        *,
        indent: str,
    ) -> str:
        value = getattr(self, f.name)
        if isinstance(value, int):
            return self._report_item(
                item=f.name,
                count=value,
                shortest_item_size=top_level_span[0],
                longest_item_size=top_level_span[1],
                indent=indent,
                indents=1,
                item_count_sep=":",
            )
        elif isinstance(value, dict):
            return self._report_plain_histogram(f.name, indent=indent)
        else:
            raise ValueError(f"Unsupported field for making report: {f}")

    def _report_plain_histogram(
        self,
        field_name: str,
        *,
        indent: str,
        indents: int = 1,
    ) -> str:
        header = self._report_header(field_name, indent=indent, indents=indents)
        item_strings = []
        plain_histogram: PlainHistogram = getattr(self, field_name)
        span = self._length_span([str(k) for k in plain_histogram])
        sorted_hist = sorted(plain_histogram.items(), key=lambda x: x[1], reverse=True)
        for item, count in sorted_hist:
            string = self._report_item(
                item=str(item),
                count=count,
                shortest_item_size=span[0],
                longest_item_size=span[1],
                indent=indent,
                indents=indents + 1,
            )
            item_strings.append(string)
        return header + "".join(item_strings)

    @staticmethod
    def _make_indent(*, indent: str, indents: int) -> str:
        return "".join(indent for _ in range(indents))

    @classmethod
    def _report_header(cls, header: Any, *, indent: str, indents: int) -> str:
        return f"{cls._make_indent(indent=indent, indents=indents)}{header}:\n"

    def _report_item(
        self,
        item: Sized,
        count: int,
        *,
        shortest_item_size: int,
        longest_item_size: int,
        indent: str,
        indents: int,
        padding_dot_threshold: int = 20,
        item_count_sep: str = "",
        max_count_digits: int = 5,
    ) -> str:
        max_padding = longest_item_size - shortest_item_size
        padding = " " if max_padding < padding_dot_threshold else "."
        padding = " " + (padding * max(0, longest_item_size - len(item))) + " "
        indent = self._make_indent(indent=indent, indents=indents)
        percent_padding = " " * (max_count_digits - len(str(count)))
        percent = f"{percent_padding}({round(100 * count / self.total_count, 1)}%)"
        return f"{indent}{item}{item_count_sep}{padding}{count}{percent}\n"

    def as_original(self, item: Datum) -> Datum:
        return self.originals.get(item, item)

    def as_originals(self, plain_histogram: PlainHistogram) -> PlainHistogram:
        return {self.as_original(k): v for k, v in plain_histogram.items()}

    @staticmethod
    def _length_span(options: Iterable[Sized]) -> tuple[int, int]:
        lengths = [len(option) for option in options] or [0]
        return min(lengths), max(lengths)

    @staticmethod
    def is_missing(item: Datum) -> bool:
        return isinstance(item, float) and math.isnan(item)

    @staticmethod
    def is_non_missing_float(item: Datum) -> bool:
        return isinstance(item, float) and not math.isnan(item)

    @staticmethod
    def is_non_empty_string(item: Datum) -> bool:
        return isinstance(item, str) and len(item.strip()) > 0

    @staticmethod
    def is_empty_string(item: Datum) -> bool:
        return isinstance(item, str) and len(item.strip()) == 0

    @staticmethod
    def _add_item(histogram: PlainHistogram, item: Datum, count: int = 1) -> None:
        histogram.setdefault(item, 0)
        histogram[item] += count

    @staticmethod
    def raise_bad_data_error(item: Datum) -> None:
        raise ValueError(f"Unsupported data type: '{item}' should be of type: {Datum}")


@dataclass
class IntervalOrDateOrRejectHistogram(Histogram):
    missing: int = 0
    rejects: int = 0
    intervals: dict[float, int] = field(default_factory=dict)
    dates: dict[str, int] = field(default_factory=dict)

    def fill_data(self, items: Data, parser_data: dict[str, Any]) -> None:
        parser = IntervalOrDateParser(**parser_data).reject
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_missing_float(item):
                self._add_item(self.intervals, item)
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, parser)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(self, item: str, parser: Optional[EnumParser]) -> None:
        try:
            self._add_item(self.intervals, float(item))
        except ValueError:
            if parser is not None and parser(item) is not None:
                self.rejects += 1
            else:
                self._add_item(self.dates, item)


@dataclass
class EnumHistogram(Histogram):
    missing: int = 0
    matches: dict[str, int] = field(default_factory=dict)

    def fill_data(self, items: Data, parser_data: dict[str, Any]) -> None:
        parser = EnumParser.from_options(**parser_data)
        for option in parser.options:
            self.matches.setdefault(option.value, 0)
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, parser)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(self, item: str, parser: EnumParser) -> None:
        enum_field = parser(item)
        if enum_field is None:
            self.raise_bad_data_error(item)
        self._add_item(self.matches, str(enum_field.value))


@dataclass
class ListOfStringsHistogram(Histogram):
    missing: int = 0
    individual_items: dict[str, int] = field(default_factory=dict)
    complete_list: dict[str, int] = field(default_factory=dict)

    def fill_data(self, items: Data, list_item_separator: str) -> None:
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, list_item_separator)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(self, item: str, list_item_separator: str) -> None:
        individual_items = []
        for individual_item in item.split(list_item_separator):
            lowered_item = individual_item.lower()
            individual_items.append(lowered_item)
            self._add_item(self.individual_items, lowered_item)
            self.originals.setdefault(lowered_item, individual_item)
        individual_items = list_item_separator.join(sorted(individual_items))
        self._add_item(self.complete_list, individual_items)
        self.originals.setdefault(individual_items, item)


@dataclass
class ListOfEnumsHistogram(Histogram):
    missing: int = 0
    individual_matches: dict[str, int] = field(default_factory=dict)
    complete_list: dict[str, int] = field(default_factory=dict)

    def fill_data(
        self,
        items: Data,
        parser_data: dict[str, Any],
        list_item_separator: str,
    ) -> None:
        histogram = ListOfStringsHistogram("", "", 0)
        histogram.fill_data(items, list_item_separator)
        self.complete_list = histogram.as_originals(histogram.complete_list)
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, parser_data, list_item_separator)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(
        self,
        item: Datum,
        parser_data: dict[str, Any],
        list_item_separator: str,
    ) -> None:
        items = item.split(list_item_separator)
        histogram = EnumHistogram("", "", 0)
        histogram.fill_data(items, parser_data)
        for processed_item, count in histogram.matches.items():
            self._add_item(self.individual_matches, processed_item, count=count)


@dataclass
class ListOrEnumsHistogram(Histogram):
    missing: int = 0
    enum_matches: dict[str, int] = field(default_factory=dict)
    individual_items: dict[str, int] = field(default_factory=dict)
    complete_list: dict[str, int] = field(default_factory=dict)

    def fill_data(
        self,
        items: Data,
        parser_data: dict[str, Any],
        list_item_separator: str,
    ) -> None:
        parser = EnumParser.from_options(**parser_data)
        for option in parser.options:
            self.enum_matches.setdefault(option.value, 0)
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, parser, list_item_separator)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(
        self,
        item: Datum,
        parser: EnumParser,
        list_item_separator: str,
    ) -> None:
        enum_field = parser(item)
        if enum_field is None:
            histogram = ListOfStringsHistogram("", "", 0)
            histogram.fill_data([item], list_item_separator)
            for processed_item, count in histogram.individual_items.items():
                # We purposefully do not want originals here.
                self._add_item(self.individual_items, processed_item, count=count)
            for processed_item, count in histogram.complete_list.items():
                # We purposefully do not want originals here.
                self._add_item(self.complete_list, processed_item, count=count)
        else:
            self._add_item(self.enum_matches, str(enum_field.value))


@dataclass
class MultiTagHistogram(Histogram):
    missing: int = 0
    tag_matches: dict[str, int] = field(default_factory=dict)
    individual_tags: dict[str, int] = field(default_factory=dict)
    complete_list: dict[str, int] = field(default_factory=dict)

    def fill_data(
        self,
        items: Data,
        parser_data: dict[str, Any],
        list_item_separator: str,
    ) -> None:
        parser = MultiTagParser.from_tag_data(**parser_data)
        for option in parser.tags:
            self.tag_matches.setdefault(option, 0)
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, list_item_separator)
            else:
                self.raise_bad_data_error(item)

    def _add_non_empty_string(self, item: Datum, list_item_separator: str) -> None:
        histogram = ListOfStringsHistogram("", "", 0)
        histogram.fill_data([item], list_item_separator)
        complete_item = []
        for processed_item, count in histogram.individual_items.items():
            processed_item = histogram.as_original(processed_item)
            tag = Tag.from_string(processed_item)
            self._add_item(self.individual_tags, tag.to_string(), count=count)
            self._add_item(self.tag_matches, tag.tag)
            complete_item.append(tag.to_string())
        count = next(iter(histogram.complete_list.values()))  # There is only 1 item.
        complete_item = list_item_separator.join(sorted(complete_item))
        self._add_item(self.complete_list, complete_item, count=count)


@dataclass
class ScoreOrRejectHistogram(Histogram):
    missing: int = 0
    rejects: int = 0
    scores: dict[float, int] = field(default_factory=dict)

    def fill_data(self, items: Data, parser_data: dict[str, Any]) -> None:
        parser = ScoreOrRejectParser(**parser_data)
        self._maybe_init_scores(parser)
        for item in items:
            if self.is_missing(item) or self.is_empty_string(item):
                self.missing += 1
            elif self.is_non_missing_float(item):
                self._add_item(self.scores, item)
            elif self.is_non_empty_string(item):
                self._add_non_empty_string(item, parser.reject)
            else:
                self.raise_bad_data_error(item)

    def _maybe_init_scores(self, p: ScoreOrRejectParser) -> None:
        if p.force_int and p.min_score is not None and p.max_score is not None:
            for i in range(int(p.min_score), int(p.max_score) + 1):
                self.scores.setdefault(i, 0)

    def _add_non_empty_string(self, item: str, parser: Optional[EnumParser]) -> None:
        try:
            self._add_item(self.scores, float(item))
        except ValueError:
            if parser is not None and parser(item) is not None:
                self.rejects += 1
            else:
                self.raise_bad_data_error(item)


class HistogramMaker:
    def __init__(self, clusters: ClustersConfig, consolidate: ConsolidateConfig):
        self.clusters = clusters
        self.consolidate = consolidate
        self.parser_types = {
            name: enum_from_str(ParserType, data.parser_type)
            for name, data in clusters.clusters.items()
        }
        self.parser_data = {
            name: data.parser_data for name, data in clusters.clusters.items()
        }

    def __call__(self, consolidation_file: str, *args, **kwargs) -> list[Histogram]:
        df = pd.read_csv(consolidation_file, index_col=False)
        histograms = []
        for name in self.clusters.included_clusters:
            name_df = df[[self.consolidate.llm_column_name, name]]
            for llm, group in name_df.groupby(by=self.consolidate.llm_column_name):
                histogram = self._make_histogram(name, llm, group[name].to_list())
                histograms.append(histogram)
        return histograms

    def _make_histogram(
        self,
        name: ClusterName,
        llm: Nickname,
        data: list[Union[str, float]],
    ) -> Histogram:
        parser_type, parser_data = self.parser_types[name], self.parser_data[name]
        if parser_type == ParserType.INTERVAL_OR_DATE_OR_REJECT:
            histogram = IntervalOrDateOrRejectHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data)
            return histogram
        elif parser_type == ParserType.ENUM:
            histogram = EnumHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data)
            return histogram
        elif parser_type == ParserType.LIST_OF_STRINGS:
            histogram = ListOfStringsHistogram(name, llm, len(data))
            histogram.fill_data(data, self.consolidate.list_item_separator)
            return histogram
        elif parser_type == ParserType.LIST_OF_ENUMS:
            histogram = ListOfEnumsHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data, self.consolidate.list_item_separator)
            return histogram
        elif parser_type == ParserType.LIST_OR_ENUM:
            histogram = ListOrEnumsHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data, self.consolidate.list_item_separator)
            return histogram
        elif parser_type == ParserType.MULTI_TAG:
            histogram = MultiTagHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data, self.consolidate.list_item_separator)
            return histogram
        elif parser_type == ParserType.SCORE_OR_REJECT:
            histogram = ScoreOrRejectHistogram(name, llm, len(data))
            histogram.fill_data(data, parser_data)
            return histogram
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")
