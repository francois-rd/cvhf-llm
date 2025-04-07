from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Optional
from enum import Enum

from ..parsing import (
    IntervalOrDateOrReject,
    ListOrEnum,
    ScoreOrReject,
    Tag,
)


class ComparisonSate(Enum):
    VALID = "VALID"
    NO_LLM = "NO_LLM"
    NO_LABEL = "NO_LABEL"
    INVALID_OTHER = "INVALID_OTHER"

    def is_valid(self) -> bool:
        return self == ComparisonSate.VALID

    def raise_if_valid(self):
        if self.is_valid():
            raise ValueError(f"Expected invalid sate. Got: {self}")


@dataclass
class Comparison:
    state: ComparisonSate

    def is_valid(self) -> bool:
        return self.state.is_valid()

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "Comparison":
        state.raise_if_valid()
        return Comparison(state=state)


@dataclass
class StringComparison(Comparison):
    exact: Optional[bool]
    agreement: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "StringComparison":
        state.raise_if_valid()
        return StringComparison(state=state, exact=None, agreement=None)

    @staticmethod
    def from_strings(llm: Optional[str], label: Optional[str]) -> "StringComparison":
        if llm is None:
            return StringComparison.from_invalid_state(ComparisonSate.NO_LLM)
        if label is None:
            return StringComparison.from_invalid_state(ComparisonSate.NO_LABEL)
        llm, label = llm.strip().upper(), label.strip().upper()
        return StringComparison(
            state=ComparisonSate.VALID,
            exact=llm == label,
            agreement=StringComparison.compute_agreement(llm, label),
        )

    @staticmethod
    def compute_agreement(llm: str, label: str) -> float:
        # TODO: If this isn't good enough, we can switch to a full
        #  semantic similarity language model approach. Expensive though.
        return SequenceMatcher(lambda x: x == " ", llm, label).ratio()


@dataclass
class EnumComparison(Comparison):
    match: Optional[bool]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "EnumComparison":
        state.raise_if_valid()
        return EnumComparison(state=state, match=None)

    @staticmethod
    def from_strings(llm: Optional[str], label: Optional[str]) -> "EnumComparison":
        result = StringComparison.from_strings(llm, label)
        if not result.is_valid():
            return EnumComparison.from_invalid_state(result.state)
        return EnumComparison(state=ComparisonSate.VALID, match=result.exact)


@dataclass
class TagComparison(Comparison):
    tag_match: Optional[bool]
    has_value: bool
    exact_value: Optional[bool]
    value_agreement: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **kwargs) -> "TagComparison":
        state.raise_if_valid()
        try:
            has_value = kwargs["has_value"]
        except KeyError:
            raise ValueError(f'Missing required keyword argument: "has_value"')
        return TagComparison(
            state=state,
            tag_match=None,
            has_value=has_value,
            exact_value=None,
            value_agreement=None,
        )

    @staticmethod
    def from_tags(llm: Optional[Tag], label: Tag) -> "TagComparison":
        has_val = label.value is not None
        from_invalid_state = TagComparison.from_invalid_state
        if llm is None:
            return from_invalid_state(ComparisonSate.NO_LLM, has_value=has_val)
        if label is None:
            return from_invalid_state(ComparisonSate.NO_LABEL, has_value=has_val)
        match = EnumComparison.from_strings(llm.tag, label.tag).match
        if llm.value is None or label.value is None:
            return TagComparison(
                state=ComparisonSate.VALID,
                tag_match=match,
                has_value=False,
                exact_value=None,
                value_agreement=None,
            )
        comp = StringComparison.from_strings(llm.value, label.value)
        return TagComparison(
            state=ComparisonSate.VALID,
            tag_match=match,
            has_value=True,
            exact_value=comp.exact,
            value_agreement=comp.agreement,
        )


@dataclass
class ListOfStringsComparison(Comparison):
    recall_exact: Optional[float]
    mean_agreement: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "ListOfStringsComparison":
        state.raise_if_valid()
        return ListOfStringsComparison(
            state=state,
            recall_exact=None,
            mean_agreement=None,
        )

    @staticmethod
    def from_comparisons(
        per_label_comparisons: list[StringComparison],
    ) -> "ListOfStringsComparison":
        if not any(c.is_valid() for c in per_label_comparisons):
            return ListOfStringsComparison.from_invalid_state(
                ComparisonSate.INVALID_OTHER,
            )
        exact = sum(c.exact for c in per_label_comparisons if c.is_valid())
        agreement = sum(c.agreement for c in per_label_comparisons if c.is_valid())
        return ListOfStringsComparison(
            state=ComparisonSate.VALID,
            recall_exact=exact / len(per_label_comparisons),
            mean_agreement=agreement / len(per_label_comparisons),
        )


@dataclass
class ListOfEnumsComparison(Comparison):
    recall_match: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "ListOfEnumsComparison":
        state.raise_if_valid()
        return ListOfEnumsComparison(state=state, recall_match=None)

    @staticmethod
    def from_comparison(comp: ListOfStringsComparison) -> "ListOfEnumsComparison":
        if not comp.is_valid():
            return ListOfEnumsComparison.from_invalid_state(comp.state)
        return ListOfEnumsComparison(
            state=ComparisonSate.VALID,
            recall_match=comp.recall_exact,
        )


@dataclass
class MultiTagComparison(Comparison):
    recall_match_tag: Optional[float]
    recall_exact_value: Optional[float]
    mean_value_agreement: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "MultiTagComparison":
        state.raise_if_valid()
        return MultiTagComparison(
            state=state,
            recall_match_tag=None,
            recall_exact_value=None,
            mean_value_agreement=None,
        )

    @staticmethod
    def from_comparisons(
        per_label_comparisons: list[TagComparison],
    ) -> "MultiTagComparison":
        if not any(c.is_valid() for c in per_label_comparisons):
            return MultiTagComparison.from_invalid_state(ComparisonSate.INVALID_OTHER)
        match = sum(c.tag_match for c in per_label_comparisons if c.is_valid())
        comps_with_value = [c for c in per_label_comparisons if c.has_value]
        if len(comps_with_value) == 0:
            return MultiTagComparison(
                state=ComparisonSate.VALID,
                recall_match_tag=match / len(per_label_comparisons),
                recall_exact_value=None,
                mean_value_agreement=None,
            )
        exact = sum(c.exact_value for c in comps_with_value if c.is_valid())
        agreement = sum(c.value_agreement for c in comps_with_value if c.is_valid())
        return MultiTagComparison(
            state=ComparisonSate.VALID,
            recall_match_tag=match / len(per_label_comparisons),
            recall_exact_value=exact / len(comps_with_value),
            mean_value_agreement=agreement / len(comps_with_value),
        )


@dataclass
class ScoreComparison(Comparison):
    exact: Optional[bool]
    difference: Optional[float]

    @staticmethod
    def from_invalid_state(state: ComparisonSate, **_) -> "ScoreComparison":
        state.raise_if_valid()
        return ScoreComparison(state=state, exact=None, difference=None)

    @staticmethod
    def from_scores(
        llm: Optional[float],
        label: Optional[float],
        tol: float,
    ) -> "ScoreComparison":
        if llm is None:
            return ScoreComparison.from_invalid_state(ComparisonSate.NO_LLM)
        if label is None:
            return ScoreComparison.from_invalid_state(ComparisonSate.NO_LABEL)
        return ScoreComparison(
            state=ComparisonSate.VALID,
            exact=abs(llm - label) < tol,
            difference=llm - label,
        )


class Comparator:
    def __call__(self, llm_output: Any, label: Any, *args, **kwargs) -> Comparison:
        raise NotImplementedError


class IntervalOrDateComparator(Comparator):
    def __init__(self, tol: float):
        self.tol = tol

    def __call__(self, llm_output: Any, label: Any, *args, **kwargs) -> Comparison:
        llm = IntervalOrDateOrReject(**llm_output)
        label = IntervalOrDateOrReject(**label)
        result = ScoreComparison.from_scores(llm.interval, label.interval, self.tol)
        if result.is_valid():
            return result
        result = StringComparison.from_strings(llm.date, label.date)
        if result.is_valid():
            return result
        result = EnumComparison.from_strings(llm.reject, label.reject)
        if result.is_valid():
            return result
        return Comparison.from_invalid_state(ComparisonSate.INVALID_OTHER)


class EnumComparator(Comparator):
    def __call__(self, llm_output: Any, label: Any, *args, **kwargs) -> EnumComparison:
        return EnumComparison.from_strings(llm_output, label)


class ListOfStringsComparator(Comparator):
    def __call__(
        self,
        llm_output: Any,
        label: Any,
        *args,
        **kwargs,
    ) -> ListOfStringsComparison:
        comparisons, used_llm = [], []
        for label_item in label:  # Label is a list[str]
            comparison, llm = self._find_best_llm(llm_output, label_item, used_llm)
            comparisons.append(comparison)
            if llm is not None:
                used_llm.append(llm)
        return ListOfStringsComparison.from_comparisons(comparisons)

    @staticmethod
    def _find_best_llm(
        llm_output: list[str],
        label: str,
        llm_blacklist: list[str],
    ) -> tuple[StringComparison, Optional[str]]:
        best_comp, best_llm = None, None
        for llm in llm_output:
            if llm in llm_blacklist:
                continue
            comparison = StringComparison.from_strings(llm, label)
            if not comparison.is_valid():
                continue
            if comparison.exact:
                comparison.agreement = 1.0
                return comparison, llm
            if best_comp is None or comparison.agreement > best_comp.agreement:
                best_comp = comparison
                best_llm = llm
        if best_comp is None:
            return StringComparison.from_invalid_state(ComparisonSate.NO_LLM), None
        return best_comp, best_llm


class ListOfEnumsComparator(Comparator):
    def __init__(self):
        self.comparator = ListOfStringsComparator()

    def __call__(
        self,
        llm_output: Any,
        label: Any,
        *args,
        **kwargs,
    ) -> ListOfEnumsComparison:
        result = self.comparator(llm_output, label, *args, **kwargs)
        return ListOfEnumsComparison.from_comparison(result)


class ListOrEnumComparator(Comparator):
    def __init__(self):
        self.comparator = ListOfStringsComparator()

    def __call__(self, llm_output: Any, label: Any, *args, **kwargs) -> Comparison:
        llm = ListOrEnum(**llm_output)
        label = ListOrEnum(**label)
        result = EnumComparison.from_strings(llm.enum_value, label.enum_value)
        if result.is_valid():
            return result
        result = self._compare_list(llm.strings, label.strings, *args, **kwargs)
        if result.is_valid():
            return result
        return Comparison.from_invalid_state(ComparisonSate.INVALID_OTHER)

    def _compare_list(
        self,
        llm: Optional[list[str]],
        label: Optional[list[str]],
        *args,
        **kwargs,
    ) -> Comparison:
        if llm is None:
            return Comparison.from_invalid_state(ComparisonSate.NO_LLM)
        if label is None:
            return Comparison.from_invalid_state(ComparisonSate.NO_LABEL)
        return self.comparator(llm, label, *args, **kwargs)


class MultiTagComparator(Comparator):
    def __call__(
        self,
        llm_output: Any,
        label: Any,
        *args,
        **kwargs,
    ) -> MultiTagComparison:
        llm_output = [Tag(**llm) for llm in llm_output]  # llm_output is list[dict]
        label = [Tag(**label_item) for label_item in label]  # label is list[dict]
        comparisons, used_llm = [], []
        for label_item in label:
            comparison, llm = self._find_best_llm(llm_output, label_item, used_llm)
            comparisons.append(comparison)
            if llm is not None:
                used_llm.append(llm)
        return MultiTagComparison.from_comparisons(comparisons)

    @staticmethod
    def _find_best_llm(
        llm_output: list[Tag],
        label: Tag,
        llm_blacklist: list[Tag],
    ) -> tuple[TagComparison, Optional[Tag]]:
        best_comp, best_llm = None, None
        for llm in llm_output:
            if llm in llm_blacklist:
                continue
            comp = TagComparison.from_tags(llm, label)
            if not comp.is_valid() or not comp.tag_match:
                continue
            if not comp.has_value:
                return comp, llm
            if comp.exact_value:
                comp.value_agreement = 1.0
                return comp, llm
            if best_comp is None or comp.value_agreement > best_comp.value_agreement:
                best_comp = comp
                best_llm = llm
        if best_comp is None:
            state = ComparisonSate.NO_LLM
            if len(llm_output) > 0:
                state = ComparisonSate.INVALID_OTHER
            has_value = label.value is not None
            best_comp = TagComparison.from_invalid_state(state, has_value=has_value)
            return best_comp, None
        return best_comp, best_llm


class ScoreOrRejectComparator(Comparator):
    def __init__(self, tol: float):
        self.tol = tol

    def __call__(self, llm_output: Any, label: Any, *args, **kwargs) -> Comparison:
        llm = ScoreOrReject(**llm_output)
        label = ScoreOrReject(**label)
        result = EnumComparison.from_strings(llm.reject, label.reject)
        if result.is_valid():
            return result
        result = ScoreComparison.from_scores(llm.score, label.score, self.tol)
        if result.is_valid():
            return result
        return Comparison.from_invalid_state(ComparisonSate.INVALID_OTHER)


def test_comparators():
    # Yes, this is not the conventional to implement tests, but we don't need an
    # entire testing infrastructure set up for this dev code.
    test_score_string_and_enum()
    test_list_of_strings()
    test_multi_tag()
    print("All tests passed.")


def test_score_string_and_enum():
    comp = IntervalOrDateComparator(0.0001)
    int_llm = IntervalOrDateOrReject(180, None, None)
    int_label = IntervalOrDateOrReject(180, None, None)
    int_exact_result = comp(asdict(int_llm), asdict(int_label))
    int_exact_expect = ScoreComparison(ComparisonSate.VALID, exact=True, difference=0)
    assert int_exact_result == int_exact_expect
    int_off_llm = IntervalOrDateOrReject(170, None, None)
    int_off_result = comp(asdict(int_off_llm), asdict(int_label))
    int_off_expect = ScoreComparison(ComparisonSate.VALID, exact=False, difference=-10)
    assert int_off_result == int_off_expect

    date_llm = IntervalOrDateOrReject(None, "2020-02-10", None)
    date_label = IntervalOrDateOrReject(None, "2020-02-10", None)
    date_exact_result = comp(asdict(date_llm), asdict(date_label))
    date_exact_expect = StringComparison(ComparisonSate.VALID, exact=True, agreement=1)
    assert date_exact_result == date_exact_expect
    date_off_llm = IntervalOrDateOrReject(None, "2019-03-10", None)
    date_off_result = comp(asdict(date_off_llm), asdict(date_label))
    a = StringComparison.compute_agreement("2019-03-10", "2020-02-10")
    date_off_expect = StringComparison(ComparisonSate.VALID, exact=False, agreement=a)
    assert date_off_result == date_off_expect

    rej_llm = IntervalOrDateOrReject(None, None, "IRRELEVANT")
    rej_label = IntervalOrDateOrReject(None, None, "IRRELEVANT")
    rej_exact_result = comp(asdict(rej_llm), asdict(rej_label))
    rej_exact_expect = EnumComparison(ComparisonSate.VALID, match=True)
    assert rej_exact_result == rej_exact_expect
    rej_off_llm = IntervalOrDateOrReject(None, None, "MAY")
    rej_off_result = comp(asdict(rej_off_llm), asdict(rej_label))
    rej_off_expect = EnumComparison(ComparisonSate.VALID, match=False)
    assert rej_off_result == rej_off_expect


def test_list_of_strings():
    comp = ListOfStringsComparator()
    blank_result = comp([], ["A", "B"])
    blank_expected = ListOfStringsComparison.from_invalid_state(
        ComparisonSate.INVALID_OTHER,
    )
    assert blank_result == blank_expected

    mismatch_result = comp(["something"], ["some overlap", "unrelated"])
    agreement = StringComparison.compute_agreement("something", "some overlap")
    mismatch_expected = ListOfStringsComparison(
        state=ComparisonSate.VALID,
        recall_exact=(0 + 0) / 2,
        mean_agreement=(agreement + 0.0) / 2,
    )
    assert mismatch_result == mismatch_expected

    one_match_result = comp(["something"], ["something", "else"])
    one_match_expected = ListOfStringsComparison(
        state=ComparisonSate.VALID,
        recall_exact=(1 + 0) / 2,
        mean_agreement=(1.0 + 0.0) / 2,
    )
    assert one_match_result == one_match_expected

    full_recall_result = comp(["something", "else"], ["something"])
    full_recall_expected = ListOfStringsComparison(
        state=ComparisonSate.VALID,
        recall_exact=1 / 1,
        mean_agreement=1.0 / 1,
    )
    assert full_recall_result == full_recall_expected

    partial_result = comp(
        ["heart failure", "heart", "short of breath"],
        ["shortness of breath", "heart failure", "palpitations"],
    )
    agreement1 = StringComparison.compute_agreement(
        "short of breath",
        "shortness of breath",
    )
    agreement2 = StringComparison.compute_agreement(
        "heart",
        "palpitations",
    )
    partial_expected = ListOfStringsComparison(
        state=ComparisonSate.VALID,
        recall_exact=1 / 3,
        mean_agreement=(1.0 + agreement1 + agreement2) / 3,
    )
    assert partial_result == partial_expected


def test_multi_tag():
    # NOTE: We know from the interface of the MultiTagParser that:
    #  1. Repeated tags when the tag is not marked repeatable fails.
    #  2. Missing a value for a tag that requires a value fails.
    #  3. Added value for a tag without a value fails.
    # So we don't need to test these cases (e.g., llm outputs "YES" to label "YES: val")
    # because they simply cannot happen (assuming no bug in the parser, which we do
    # assume for this unit test).
    comp = MultiTagComparator()

    blank_result = comp([], [asdict(Tag(tag="A", value=None))])
    blank_expected = MultiTagComparison.from_invalid_state(ComparisonSate.INVALID_OTHER)
    assert blank_result == blank_expected

    mismatch_llm = [asdict(Tag("something", None))]
    mismatch_label = [asdict(Tag("some overlap", None)), asdict(Tag("unrelated", None))]
    mismatch_result = comp(mismatch_llm, mismatch_label)
    mismatch_expected = MultiTagComparison.from_invalid_state(
        ComparisonSate.INVALID_OTHER,
    )
    assert mismatch_result == mismatch_expected

    one_match_no_val_llm = [asdict(Tag("something", None))]
    one_match_no_val_label = [asdict(Tag("something", None)), asdict(Tag("else", None))]
    one_match_no_val_result = comp(one_match_no_val_llm, one_match_no_val_label)
    one_match_no_val_expected = MultiTagComparison(
        state=ComparisonSate.VALID,
        recall_match_tag=(1 + 0) / 2,
        recall_exact_value=None,
        mean_value_agreement=None,
    )
    assert one_match_no_val_result == one_match_no_val_expected

    one_match_one_val_llm = [asdict(Tag("something", "value"))]
    one_match_one_val_label = [
        asdict(Tag("something", "value")),
        asdict(Tag("else", None)),
    ]
    one_match_one_val_result = comp(one_match_one_val_llm, one_match_one_val_label)
    one_match_one_val_expected = MultiTagComparison(
        state=ComparisonSate.VALID,
        recall_match_tag=(1 + 0) / 2,
        recall_exact_value=1 / 1,  # Tag="else" does not contribute to count.
        mean_value_agreement=1.0 / 1,  # Tag="else" does not contribute to count.
    )
    assert one_match_one_val_result == one_match_one_val_expected

    one_match_two_val_llm = [asdict(Tag("something", "value"))]
    one_match_two_val_label = [
        asdict(Tag("something", "value")),
        asdict(Tag("else", "other")),
    ]
    one_match_two_val_result = comp(one_match_two_val_llm, one_match_two_val_label)
    one_match_two_val_expected = MultiTagComparison(
        state=ComparisonSate.VALID,
        recall_match_tag=(1 + 0) / 2,
        recall_exact_value=1 / 2,
        mean_value_agreement=1.0 / 2,
    )
    assert one_match_two_val_result == one_match_two_val_expected

    full_recall_llm = [
        asdict(Tag("something", "value")),
        asdict(Tag("else", "other")),
    ]
    full_recall_label = [asdict(Tag("something", "value"))]
    full_recall_result = comp(full_recall_llm, full_recall_label)
    full_recall_expected = MultiTagComparison(
        state=ComparisonSate.VALID,
        recall_match_tag=1 / 1,
        recall_exact_value=1 / 1,
        mean_value_agreement=1.0 / 1,
    )
    assert full_recall_result == full_recall_expected

    partial_value_llm = [
        asdict(Tag("something", "val")),
        asdict(Tag("else", "other")),
    ]
    partial_value_label = [
        asdict(Tag("something", "value")),
        asdict(Tag("else", "other")),
    ]
    partial_value_result = comp(partial_value_llm, partial_value_label)
    agreement = StringComparison.compute_agreement("val", "value")
    partial_value_expected = MultiTagComparison(
        state=ComparisonSate.VALID,
        recall_match_tag=(1 + 1) / 2,
        recall_exact_value=(0 + 1) / 2,
        mean_value_agreement=(agreement + 1.0) / 2,
    )
    assert partial_value_result == partial_value_expected
