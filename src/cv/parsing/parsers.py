from dataclasses import dataclass
from typing import Optional
import datetime

from .base import (
    EnumParser,
    FloatMatchParser,
    ListOfStringsParser,
    OutputParser,
    RegexExtractionParser,
    RegexMatchParser,
    Tag,
    TagParser,
)


@dataclass
class ListOrEnum:
    """Output of a ListOrEnumParser."""

    # The extracted list of string, if any.
    strings: Optional[list[str]]

    # The value of the extracted Enum, if any.
    enum_value: Optional[str]


class ListOrEnumParser(OutputParser):
    def __init__(
        self,
        enum_options: list[str] = None,
        sep: str = ",",
        strip: bool = True,
    ):
        """
        Parses the LLM output along one of two primary options:
        1. The LLM outputs a list of strings.
        2. The LLM outputs an Enum.

        :param enum_options: List of valid Enum options the LLM might output.
        :param sep: If the LLM is outputting a list of strings instead of an Enum,
            'sep' represents the string separator between string items.
        :param strip: If the LLM is outputting a list of strings instead of an Enum,
            whether to str.strip() each list item before returning.
        """
        super().__init__()
        if enum_options is None:
            self.enum_parser = None
        else:
            self.enum_parser = EnumParser.from_options(enum_options)
        self.list_parser = ListOfStringsParser(sep=sep, strip=strip)

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[ListOrEnum]:
        """Returns a ListOrEnum or None if neither option can be parsed."""
        if self.enum_parser is None:
            result = None
        else:
            result = self.enum_parser(generated_text, *args, **kwargs)

        if result is None:
            result = self.list_parser(generated_text, *args, **kwargs)
            if result is None:
                return None
            return ListOrEnum(strings=result, enum_value=None)
        return ListOrEnum(strings=None, enum_value=str(result.value))


class ListOfEnumsParser(OutputParser):
    def __init__(self, enum_options: list[str], sep: str = ","):
        """
        Parses the LLM output into a list, where each list item is an Enum option.

        :param enum_options: List of valid Enum options the LLM can output.
        :param sep: The string separator between Enum items in the list.
        """
        super().__init__()
        self.enum_parser = EnumParser.from_options(enum_options)
        self.list_parser = ListOfStringsParser(sep=sep)

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[list[str]]:
        result = self.list_parser(generated_text, *args, **kwargs)
        if result is None:
            return None

        output = []
        for item in result:
            enum_result = self.enum_parser(item, *args, **kwargs)
            if enum_result is None:
                return None
            output.append(str(enum_result.value))
        return output


@dataclass
class IntervalOrDateOrReject:
    """
    Output of a IntervalOrDateParser.

    NOTE: Rejection means the LLM outputted a rejection string instead of an
    interval or date. This differs from a None field, since only 1 field can
    ever be non-None (mutual exclusivity).
    """

    # A short interval of time, given in minutes, as computed from the LLM's output.
    interval: Optional[float]

    # The date outputted by the LLM in ISO format (YYYY-MM-DD). More precisely,
    # this is either the absolute date outputted by the LLM or it is an absolute
    # date derived from computing the relative difference between a reference
    # date and the relative date outputted by the LLM.
    date: Optional[str]

    # The rejection string that the LLM outputted instead of an interval or date.
    reject: Optional[str]


class IntervalOrDateParser(OutputParser):
    def __init__(self, reject: Optional[str] = None):
        """
        Parses the LLM output into a short interval, an absolute date, or a rejection.

        The short interval results from a parse of a time interval measured in minutes
        or hours, and the result is always given in minutes. For example, a parse of
        "10 hours" results in "600".

        The absolute date can result from a direct parse of an absolute date, such as
        "1995-10-13", but can also result from the parse of a long time interval
        (anything measured in units longer than hours) together with a reference
        date. For example, a parse of "6 months" (implicitly before) a reference
        date of 1995-10-13 results in "1995-04-16".

        In detail, the parse can be:
        1. "DIGIT UNIT", where DIGIT must be a valid integer and UNIT must be one of:
                minute(s), hour(s), day(s), week(s), month(s), year(s)
            where the trailing "s" is optional.
            NOTE: Only 'minutes' and 'hours' result in short time intervals. See below.
        2. "YYYY-MM-DD", representing a valid date in ISO format.
        3. "MONTH", representing a valid month name (string), either in full or as
            a three-letter abbreviation (e.g., "January" or "Jan").
        4. "YYYY", representing a valid year.
        5. "REJECT", which, if given (not None) allows the LLM to output this
            specific string instead of providing an interval or date format.

        For the "DIGIT UNIT", "MONTH", and "YYYY" parses, the interpretation is
        that these are relative time intervals in the past (e.g., "3 months" means
        "3 months ago"; "Feb" means "last February"; "1995" means "in 1995"). To
        convert these into absolute dates, the __call__() method accepts a
        "reference_date" keyword argument from which the relative time interval
        will be subtracted. This value is provided in __call__() rather than in
        __init__() to allow that the reference date might vary between calls.
        However, note that whenever "UNIT" is one of ['minutes', 'hours'], the
        interpretation switches to a short time interval representing a duration
        of an event (e.g., "my symptoms typically last around 10 minutes").

        Whenever the date is imprecise (e.g., "MONTH" and "YYYY"), we assume the
        midpoint of the given timeframe and compute dates relative to that. For
        example, "Feb" relative to a reference date of 2020-03-23 results in
        2020-02-15, because 15 is assumed to be the midpoint of a month (on average).
        Similarly, the midpoint of a year is assumed to be July 1st.

        If "MONTH" occurs earlier in the year than the reference date month, then
        it is assumed to be the same year. For example, "Feb" relative to a reference
        date of 2020-03-23 results in 2020-02-15, because February occurs earlier
        than March. If "MONTH" occurs later in the year, it is assumed to be that
        month of the previous year (assume a past event rather than a future event).
        For example, "June" relative to 2020-03-23 results in 2019-06-15, because
        June occurs later than March, so it is assumed to be June of last year.

        :param reject: The string the LLM can generate to reject answering, or
                       None if the LLM should not be able to reject.
        """
        super().__init__()
        self.interval = RegexMatchParser(
            r"(\d+)\s*(minutes?|hours?|days?|weeks?|months?|years?)",
        )
        self.iso_date = RegexExtractionParser(r"\d{4}-\d{2}-\d{2}", match_group=0)
        self.month = RegexExtractionParser(
            r"("
            r"Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|June?"
            r"|"
            r"July?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?"
            r")",
        )
        self.year = RegexExtractionParser(r"\d{4}", match_group=0)
        self.reject = None if reject is None else EnumParser.from_options([reject])

    def _output(
        self,
        interval: Optional[float] = None,
        date: Optional[datetime.date] = None,
        reject: Optional[str] = None,
    ) -> IntervalOrDateOrReject:
        if interval is not None and date is None and reject is None:
            return IntervalOrDateOrReject(interval=interval, date=None, reject=None)
        if date is not None and interval is None and reject is None:
            date = date.strftime("%Y-%m-%d")
            return IntervalOrDateOrReject(interval=None, date=date, reject=None)
        if reject is not None and interval is None and date is None:
            return IntervalOrDateOrReject(interval=None, date=None, reject=reject)
        raise ValueError(
            f"{self.__class__.__name__}: Internal error: "
            f"Lacking mutual exclusivity in output construction."
        )

    def __call__(
        self,
        generated_text: str,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        """
        Returns a 'IntervalOrDateOrReject' or None on failure to parse.
        :param kwargs: Must include a 'reference_date' keyword argument.
        """
        try:
            reference = kwargs["reference_date"]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__}: Missing 'reference_date' kwarg.",
            )
        return (
            self._maybe_attempt_reject(generated_text, *args, **kwargs)
            or self._attempt_interval(generated_text, reference, *args, **kwargs)
            or self._attempt_iso_date(generated_text, *args, **kwargs)
            or self._attempt_month(generated_text, reference, *args, **kwargs)
            or self._attempt_year(generated_text, *args, **kwargs)
            or None
        )

    def _maybe_attempt_reject(
        self,
        generated_text: str,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        if self.reject is None:
            return None
        result = self.reject(generated_text, *args, **kwargs)
        return None if result is None else self._output(reject=result)

    def _attempt_interval(
        self,
        generated_text: str,
        reference: datetime.date,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        match = self.interval(generated_text, *args, **kwargs)
        if match is None:
            return None

        try:
            digit = int(match.group(1))
        except ValueError:
            raise ValueError(
                f"{self.__class__.__name__}: Internal error: "
                f"Interval Regex could not coerce digit string to int."
            )

        unit = match.group(2)
        if not unit.endswith("s"):
            unit += "s"

        if unit == "months":
            digit *= 30
            unit = "days"

        if unit == "years":
            digit *= 365
            unit = "days"

        if unit == "minutes":
            return self._output(interval=digit)
        elif unit == "hours":
            return self._output(interval=digit * 60)
        else:
            return self._output(date=reference - datetime.timedelta(**{unit: digit}))

    def _attempt_iso_date(
        self,
        generated_text: str,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        result = self.iso_date(generated_text, *args, *kwargs)
        if result is None:
            return None
        try:
            return self._output(date=datetime.date.fromisoformat(result))
        except ValueError:
            # Assuming the 'iso_date' regex has no mistake, this means the
            # extracted date is invalid (e.g., asking for 31 days in Feb).
            return None

    def _attempt_month(
        self,
        generated_text: str,
        reference: datetime.date,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        result = self.month(generated_text, *args, **kwargs)
        if result is None:
            return None

        months_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        try:
            month = months_map[result[:3].lower()]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__}: Internal error: "
                f"Interval Regex could not coerce string into a valid month code."
            )

        output_date = datetime.date(reference.year, month, reference.day)
        if reference < output_date:
            output_date = datetime.date(reference.year - 1, month, reference.day)
        return self._output(date=output_date)

    def _attempt_year(
        self,
        generated_text: str,
        *args,
        **kwargs,
    ) -> Optional[IntervalOrDateOrReject]:
        result = self.year(generated_text, *args, **kwargs)
        if result is None:
            return None
        try:
            return self._output(date=datetime.date(int(result), 7, 1))
        except ValueError:
            return None


@dataclass
class TagData:
    """A list of these is the input to a MultiTagParser."""

    # The string name of this tag.
    tag: str

    # Whether this tag can appear repeatedly in the LLM output.
    repeatable: bool = False

    # If not None, the tag is assumed to take a mandatory value, where
    # 'value_sep' is the separator between the tag name and said value.
    # If None, the tag is assumed to never take any value.
    value_sep: Optional[str] = None


class MultiTagParser(OutputParser):
    default_sep: str = "\n"

    def __init__(
        self,
        options: list[TagData],
        sep: Optional[str] = default_sep,
    ):
        """
        Parses the LLM output according to a set of TagParsers instantiated from
        the given options.

        :param options: The detailed information on each valid tag option.
        :param sep: If not None, the LLM output is split according to this separator.
            Each resulting split must contain exactly one valid tag. If None, the
            entire LLM output must consist of exactly one valid tag.
        """
        super().__init__()
        self.tags = {tag.tag: tag for tag in options}
        self.sep = sep
        self.parsers = [TagParser(tag.tag, tag.value_sep) for tag in options]

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[list[Tag]]:
        """
        Returns a list of Tags or None. Note that 'sep' (if given) controls whether
        text splits in 'generated_text' are attempted or not.

        None is returned if parsing failed. Parsing fails if:
        - No Tags are found in 'generated_text'
        - If a non-repeatable Tag is found more than once
        - If *anything* in 'generated_text' cannot be parsed to a valid Tag
          (after splitting 'generated_text' into a list if 'sep' is given)
        - If 'sep' is given, if any item in the 'generated_text' list parses
          to more than one valid Tag (e.g., if sep=',' and generated_text=
          'TAG1 TAG2,TAG3', then parsing fails because list item 'TAG1 TAG2'
          contains more than one Tag). This prevents ambiguity with Tags that
          take a value (e.g., is 'TAG1: some value TAG2' to be parsed as
          ['TAG1: some value', 'TAG2'] or as ['TAG1: some value TAG2'] where
          'TAG2' is interpreted as part of the value of TAG1?).
        """
        all_tags = []
        found_tags = self._find_all_tags(generated_text, *args, **kwargs)
        for found_tag_name, found_tag_list in found_tags.items():
            if not self.tags[found_tag_name].repeatable and len(found_tag_list) > 1:
                return None
            all_tags.extend(found_tag_list)
        return all_tags or None

    def _find_all_tags(self, text: str, *args, **kwargs) -> dict[str, list[Tag]]:
        """
        Implementation is very harsh: if ANYTHING in the text cannot be parsed
        to a valid tag, then the entire text is rejected (no partial match on
        some tags but not others allowed).
        """
        tags = {}
        splits = [text] if self.sep is None else text.split(self.sep)
        for line in splits:
            tag = self._find_tag(line, *args, **kwargs)
            if tag is None:
                return {}
            tags.setdefault(tag.tag, []).append(tag)
        return tags

    def _find_tag(self, split: str, *args, **kwargs) -> Optional[Tag]:
        final_result = None
        for parser in self.parsers:
            result = parser(split, *args, *kwargs)
            if result is None:
                continue
            if final_result is None:
                final_result = result
            else:
                # This means there is more than one matching parse on this split.
                return None
        return final_result

    @classmethod
    def from_tag_data(cls, **kwargs) -> "MultiTagParser":
        try:
            tag_data = kwargs["options"]
        except KeyError:
            raise ValueError(
                f"{cls.__name__}: To instantiate from tag data, the input dictionary "
                f"must adhere to the following schema:\n"
                "{\n"
                '   "separator": <string>,  # This field is optional. '
                "Newline is used by default.\n"
                '   "options": [\n'
                "       {\n"
                '           "tag": <string>,\n'
                '           "repeatable": <bool>,\n'
                '           "value_sep": <string if mandatory or null for no value>,\n'
                "       },\n"
                "       ...\n"
                "   ]\n"
                "}"
            )
        sep = kwargs.get("separator", cls.default_sep)
        tags = [TagData(**data) for data in tag_data]
        return MultiTagParser(options=tags, sep=sep)


@dataclass
class ScoreOrReject:
    """
    Output of a parser that returns a Score or else detects a rejection.

    NOTE: Rejection means the LLM outputted a rejection string instead of a score.
    This differs from a None field (see each field below).
    """

    # The score outputted by the LLM.
    # This field is None only if the LLM output is a rejection instead.
    score: Optional[float]

    # The rejection string that the LLM outputted instead of a score.
    # This field is None only if the LLM output is a score instead.
    reject: Optional[str]


class ScoreOrRejectParser(OutputParser):
    def __init__(
        self,
        min_score: Optional[float],
        max_score: Optional[float],
        force_int: bool = False,
        int_tol: float = 0.0001,
        reject: Optional[str] = None,
    ):
        """
        Parses the LLM output into a score or (optionally) a rejection.

        :param min_score: Minimum allowed score, or None if there is no minimum
        :param max_score: Maximum allowed score, or None if there is no maximum
        :param force_int: Whether to fail if the score cannot be coerced to an integer.
        :param int_tol: Tolerance (away from an integer) for a score to be
                        considered a valid integer.
        :param reject: String the LLM can generate to reject answering, or None if
                       the LLM should not be able to reject.
        """
        super().__init__()
        self.min_score, self.max_score = min_score, max_score
        self.force_int, self.int_tol = force_int, int_tol
        self.reject = None if reject is None else EnumParser.from_options([reject])
        self.parsers = [
            FloatMatchParser(),
            RegexExtractionParser(r"\d+(\.\d+)?", match_group=0),
        ]

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[ScoreOrReject]:
        """
        Returns a ScoreOrReject or None.

        None is returned if parsing failed or if the score does not meet the
        scoring criteria (e.g., from "min_score" or "force_int").
        """
        if self.reject is not None:
            result = self.reject(generated_text, *args, **kwargs)
            if result is not None:
                return ScoreOrReject(score=None, reject=result)
        result = self._attempt_parse(generated_text, *args, **kwargs)
        if result is not None:
            return ScoreOrReject(score=result, reject=None)
        return None

    def _attempt_parse(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        score = None
        for parser in self.parsers:
            score = parser(generated_text, *args, **kwargs)
            try:
                score = float(score)
                break
            except (TypeError, ValueError):
                pass

        if score is None:
            return None
        if self.min_score is not None and score < self.min_score:
            return None
        if self.max_score is not None and score > self.max_score:
            return None

        if self.force_int:
            if abs(int(score) - score) < self.int_tol:
                return float(int(score))
            else:
                return None

        return score
