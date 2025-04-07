from typing import Any
from enum import Enum

from .base import EnumParser, ListOfStringsParser, OutputParser
from .parsers import (
    IntervalOrDateParser,
    ListOfEnumsParser,
    ListOrEnumParser,
    MultiTagParser,
    ScoreOrRejectParser,
)
from ..core import ClusterName, ClustersConfig
from ..io import enum_from_str


class ParserType(Enum):
    """Enumeration of all managed parser types."""

    INTERVAL_OR_DATE_OR_REJECT = "INTERVAL_OR_DATE_OR_REJECT"
    ENUM = "ENUM"
    LIST_OF_STRINGS = "LIST_OF_STRINGS"
    LIST_OF_ENUMS = "LIST_OF_ENUMS"
    LIST_OR_ENUM = "LIST_OR_ENUM"
    MULTI_TAG = "MULTI_TAG"
    SCORE_OR_REJECT = "SCORE_OR_REJECT"


def parser_factory(parser_type: str, parser_data: dict[str, Any]) -> OutputParser:
    parser_type = enum_from_str(ParserType, parser_type)
    if parser_type == ParserType.INTERVAL_OR_DATE_OR_REJECT:
        return IntervalOrDateParser(**parser_data)
    elif parser_type == ParserType.ENUM:
        return EnumParser.from_options(**parser_data)
    elif parser_type == ParserType.LIST_OF_STRINGS:
        return ListOfStringsParser(**parser_data)
    elif parser_type == ParserType.LIST_OF_ENUMS:
        return ListOfEnumsParser(**parser_data)
    elif parser_type == ParserType.LIST_OR_ENUM:
        return ListOrEnumParser(**parser_data)
    elif parser_type == ParserType.MULTI_TAG:
        return MultiTagParser.from_tag_data(**parser_data)
    elif parser_type == ParserType.SCORE_OR_REJECT:
        return ScoreOrRejectParser(**parser_data)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")


class ParserManager:
    def __init__(self, cfg: ClustersConfig):
        self.parsers = {}
        for name, data in cfg.clusters.items():
            self.parsers[name] = parser_factory(data.parser_type, data.parser_data)

    def get(self, name: ClusterName) -> OutputParser:
        try:
            return self.parsers[name]
        except KeyError:
            raise ValueError(f"Missing parser data for cluster '{name}'.")
