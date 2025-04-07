from typing import Optional
import sys
import os
import re

from docx import Document


class Tagger:
    def __init__(self, regex: str = r"^(Answer\s*(to))?\s*Question\s*(.+?)\s*(Ite(ra|ar)tion.+?)?\.\."):
        self.pattern = re.compile(regex, flags=re.IGNORECASE)
        self.digits = re.compile(r"([0-9]+)\s*\w?\s*[0-9]*")
        self.histogram = {}
        self.conversion = {}

    def __call__(self, a_id: str, lines: list[str]):
        for line in lines:
            res = self.match(line)
            if res is not None:
                if res in self.histogram:
                    self.histogram[res].append(a_id)
                else:
                    self.histogram[res] = [a_id]
                    self.conversion[res] = self.convert(res)

    def match(self, line) -> Optional[tuple]:
        match = self.pattern.search(line)
        if match is None:
            return None
        return match.groups()

    def convert(self, match_result):
        middle = match_result[1]
        try:
            return int(middle)
        except ValueError:
            res = self.digits.findall(middle)
            if len(set(res)) == 1:  # All elements equal and there is at least one.
                return int(res[0])
            return res

    def dump(self, output_file: str):
        self.histogram = dict(sorted(self.histogram.items(), key=lambda x: x[0][1]))
        str_objs = [
            f"{k}: {v} -> {self.conversion[k]}" + os.linesep
            for k, v in self.histogram.items()
            if isinstance(self.conversion[k], list) and len(v) < 5
        ]
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(str_objs)


def load_docx(file_path: str) -> list[str]:
    lines = [paragraph.text for paragraph in Document(file_path).paragraphs]
    return [line.strip() for line in lines if line.strip()]


def compute_histogram(docx_transcript_dir: str, output_file: str):
    tagger = Tagger()
    for root, _, files in os.walk(docx_transcript_dir):
        for filename in files:
            a_id = filename.split("_")[0]
            tagger(a_id, load_docx(str(os.path.join(root, filename))))
    tagger.dump(output_file)


if __name__ == "__main__":
    compute_histogram(sys.argv[1], sys.argv[2])
