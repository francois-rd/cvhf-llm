from typing import Iterable, Optional
import re

from .base import Span, Tag, TagsConfig, TagType, Transcript
from ..core import Cluster, ClustersConfig, Lines, QuestionId


class Tagger:
    def __init__(self, tags_cfg: TagsConfig):
        self.tags_cfg = tags_cfg
        self.pattern = re.compile(tags_cfg.primary_regex, flags=re.IGNORECASE)
        self.q_id = re.compile(tags_cfg.question_id_regex, flags=re.IGNORECASE)

    def __call__(self, lines: Lines, *args, **kwargs) -> list[Optional[Tag]]:
        return [self._do_tag(line) for line in lines]

    def _do_tag(self, line: str) -> Optional[Tag]:
        if self._is_header_tag(line):
            return Tag([], TagType.HEADER, line)
        match = self.pattern.search(line)
        if match is None:
            return None
        question_tag = match.group(self.tags_cfg.question_group)
        q_ids = set(self.q_id.findall(question_tag))
        q_ids = [q_id.replace(" ", "") for q_id in q_ids]
        tag_type = TagType.ANSWER if self._is_answer_tag(match) else TagType.QUESTION
        return Tag(q_ids, tag_type, match.group())

    def _is_header_tag(self, line: str) -> bool:
        for header in self.tags_cfg.headers:
            if line.lower().strip() == header.lower().strip():
                return True
        return False

    def _is_answer_tag(self, match: re.Match) -> bool:
        should_check_answer_group = self.tags_cfg.answer_to_group > 0
        if should_check_answer_group:
            answer_tag = match.group(self.tags_cfg.answer_to_group)
            return answer_tag is not None
        return False


class ConvertTagsToTranscript:
    """
    Several big assumptions:
    1. Lines within a single question have a consecutive span, as opposed to being
       fragmented across the transcript.
        -> We know this is not quite true given the presence of "Answer to" tags,
           but it is a rare edge case that is far too complex to handle at this time.
    2. Clusters can have non-consecutive questions (e.g., Q1, Q6, Q7), but the
       resulting 'cluster.lines' will contain lines in ascending order regardless of
       the listed order of 'cluster.questions'.
    3. If a tag identifies more than one question for a single line of transcript,
       then that line is repeated across each cluster to which the question belongs.
       In other words, mix-tag lines don't have to belong to the same cluster.
       Instead, their content is replicated across clusters. For example, a tag
       like "7 + 7a" on a single line in the transcript means that both "7" and "7a"
       will be included in any cluster that calls for either one in isolation.
    """

    def __init__(self, clusters_cfg: ClustersConfig):
        self.clusters_cfg = clusters_cfg
        all_data = clusters_cfg.clusters.values()
        self.unique_q_ids = {q_id for data in all_data for q_id in data.question_ids}

    def __call__(
        self,
        lines: Lines,
        tags: list[Optional[Tag]],
        *args,
        **kwargs,
    ) -> Transcript:
        transcript = {}
        all_spans = self._find_all_spans(tags)
        lines = [self._remove_tag(line, tag) for line, tag in zip(lines, tags)]
        for name, data in self.clusters_cfg.clusters.items():
            new_lines = self._find_cluster_lines(data.question_ids, lines, all_spans)
            transcript[name] = Cluster(lines=new_lines) if new_lines else None
        return Transcript(transcript)

    @staticmethod
    def _remove_tag(line: str, tag: Optional[Tag]) -> str:
        """Removes the tag sub-string from the line (if any)."""
        return line if tag is None else line.replace(tag.match_string, "").strip()

    def _find_all_spans(self, tags: list[Optional[Tag]]) -> dict[QuestionId, Span]:
        """Returns a mapping between each QuestionID and its Span based on the tags."""
        return {q_id: self._find_span(q_id, tags) for q_id in self.unique_q_ids}

    @staticmethod
    def _find_span(q_id: QuestionId, tags: list[Optional[Tag]]) -> Span:
        """Finds the span of the given QuestionID from amongst the tags."""
        start, end = None, None
        for i, tag in enumerate(tags):
            # NOTE: Current implementation treats answer tags as blanks.
            if tag is None or tag.tag_type == TagType.ANSWER:  # If we have a blank...
                # ... increment the end tag only if we've found the start already.
                # If we haven't found the start, we just skip over.
                end = None if start is None else i
                continue  # Importantly, go to next loop.

            if q_id not in tag.question_ids:  # Tag isn't blank, but q_ids not in it...
                if start is None and end is None:
                    continue  # ... continue if we haven't hit the span at all yet...
                break  # ... otherwise, we are done searching.

            # If q_id IS IN tag, then we set the start (if it's the first time we
            # hit a tag with q_id), or we increment the end (if start has been set).
            if start is None:
                start = i
            else:
                end = i
        return start, end

    def _find_cluster_lines(
        self,
        question_ids: list[QuestionId],
        lines: Lines,
        all_spans: dict[QuestionId, Span],
    ) -> Lines:
        """Returns the cluster's lines based on aggregating its question spans."""
        cluster_lines, spans = [], self._find_cluster_spans(question_ids, all_spans)
        for span in self._merge_and_sort_overlapping_spans(spans):
            cluster_lines.extend(lines[span[0] : span[1] + 1])
        return cluster_lines

    @staticmethod
    def _find_cluster_spans(
        question_ids: list[QuestionId],
        all_spans: dict[QuestionId, Span],
    ) -> list[Span]:
        """Returns all valid spans for the given question IDs from all the spans."""
        spans = []
        for q_id in question_ids:
            start, end = all_spans[q_id]
            if start is not None and end is not None:
                spans.append((start, end))
        return spans

    @staticmethod
    def _merge_and_sort_overlapping_spans(spans: list[Span]) -> Iterable[Span]:
        """
        Merges overlapping (and adjacent) spans. Yields the merged spans *in order*.
        """
        # NOTE: Implementation borrows from:
        #  https://codereview.stackexchange.com/questions/21307/consolidate-list-of-ranges-that-overlap
        spans = iter(sorted(spans))
        try:
            # The 'try' fails if 'spans' starts off empty.
            merged_start, merged_end = next(spans)
        except StopIteration:
            # Since this is a generator, return.
            # In Py>=3.7, do NOT raise StopIteration.
            return

        # NOTE: The below works *only* because the spans are sorted above.
        for start, end in spans:
            # If the new start is larger than the current end, there is a gap.
            if start > merged_end:
                # Yield the current merged span and start a new one.
                yield merged_start, merged_end
                merged_start, merged_end = start, end
            else:
                # Otherwise, the spans are either adjacent or fully overlapping. Merge.
                merged_end = max(merged_end, end)
        # As we fall off the end of the loop, yield the very last merged span.
        yield merged_start, merged_end
