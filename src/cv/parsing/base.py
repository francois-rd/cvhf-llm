from typing import Any, Optional, Type
from dataclasses import dataclass
from enum import Enum
import json
import re

from ..io import EnumSubType, enum_from_str


def re_compile(pattern: str, flags=None) -> re.Pattern:
    return re.compile(pattern) if flags is None else re.compile(pattern, flags=flags)


class OutputParser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[Any]:
        """
        Parses the generated_text of an LLM to extract some meaningful result.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class ScoreOutputParser(OutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        """
        Parses the generated_text of an LLM to extract a numerical score.
        This score can represent a binary indicator (0 vs 1), a Likert Scale
        with a given range, or an unbounded quantity, for example.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class StringOutputParser(OutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[str]:
        """
        Parses the generated_text of an LLM to extract a string result.
        This score can represent the label of an answer choice, or some
        unbounded free text, for example. Returns None on parsing failure.
        """
        raise NotImplementedError


class RegexMatchParser(StringOutputParser):
    def __init__(self, pattern: str, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a re.Match or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[re.Match]:
        return self.pattern.search(generated_text)


class RegexExtractionParser(StringOutputParser):
    def __init__(self, pattern: str, match_group: int = 1, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a string result from
        the match group or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param match_group: The group index of result within the pattern Match object.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.parser = RegexMatchParser(pattern, flags=flags)
        self.match_group = match_group

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[str]:
        match = self.parser(generated_text, *args, **kwargs)
        try:
            return match.group(self.match_group)
        except (AttributeError, ValueError):
            return None


class EnumParser(OutputParser):
    scrub_pattern: re.Pattern = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    def __init__(self, options: Type[EnumSubType]):
        """
        Returns the Enum option that the generated_text matches exactly
        (barring whitespace) or None if the text is not an exact match.
        """
        super().__init__()
        self.options = options

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[EnumSubType]:
        try:
            return enum_from_str(self.options, generated_text.strip())
        except ValueError:
            for option in self.options:
                if option.value.upper() == generated_text.strip().upper():
                    return option
            return None

    @classmethod
    def from_options(cls, enum_options: list[str]) -> "EnumParser":
        return EnumParser(Enum("Options", {cls._scrub(s): s for s in enum_options}))

    @classmethod
    def _scrub(cls, string: str) -> str:
        return "_".join(cls.scrub_pattern.findall(string))


@dataclass
class Tag:
    """Output of a TagParser."""

    # The string name of this tag.
    tag: str

    # The optional value attached to this tag.
    value: Optional[str]

    def to_string(self, sep: str = ": ") -> str:
        if self.value is None:
            return self.tag
        else:
            return f"{self.tag}{sep}{self.value.lower()}"

    @staticmethod
    def from_string(string: str, sep: str = ": ") -> "Tag":
        split = string.split(sep=sep, maxsplit=1)
        if len(split) == 1:
            return Tag(split[0], None)
        elif len(split) == 2:
            return Tag(*split)
        else:
            raise ValueError(f"Unsupported tag string format: {string}")


class TagParser(OutputParser):
    def __init__(self, tag: str, value_sep: Optional[str] = None, flags=re.IGNORECASE):
        """
        Parses LLM output based on Tag data. Returns a Tag or None on failure.

        :param tag: The string name of this tag (to search for in the LLM output).
        :param value_sep: If given, the tag is assumed to take a mandatory value,
            where 'value_sep' is the separator between the tag name and said value.
            If None, the tag is assumed to never take any value.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.tag_parser = RegexExtractionParser(tag, match_group=0, flags=flags)
        self.value_sep = value_sep

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[Tag]:
        maybe_tag, value = generated_text, None
        if self.value_sep is not None:
            split = generated_text.split(self.value_sep)
            if len(split) != 2:
                return None
            maybe_tag, value = split[0], split[1].strip()
        result = self.tag_parser(maybe_tag, *args, **kwargs)
        if result is None:
            return None
        return Tag(tag=result, value=value)


class JSONParser(StringOutputParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a score. Returns None on failure.

        :param schema_key: The key into the JSON object containing the score.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param flags: Flags to pass to re.findall(), if any.
        """
        super().__init__()
        self.schema_key = schema_key
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[str]:
        for string in [generated_text, *self.pattern.findall(generated_text)]:
            try:
                return json.loads(string, **kwargs)[self.schema_key]
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                json.decoder.JSONDecodeError,
            ):
                continue
        return None


class FloatMatchParser(ScoreOutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        """
        Returns the exactly matching number in the generated_text (barring whitespace)
        as a score or None if the text is not an exact match for a number.
        """
        try:
            return float(generated_text.strip())
        except ValueError:
            return None


class ListOfStringsParser(OutputParser):
    def __init__(self, sep: str = ",", strip: bool = True):
        """
        Uses 'sep' to split the generated_text into a list of strings.

        :param sep: The string seperator to use.
        :param strip: Whether to remove whitespace from each string item.
        """
        super().__init__()
        self.sep = sep
        self.strip = strip

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[list[str]]:
        """Returns a list of strings, or None if the list is empty."""
        items = generated_text.split(self.sep)
        if self.strip:
            return [item.strip() for item in items]
        return items or None
