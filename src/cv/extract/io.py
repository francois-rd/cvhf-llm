from typing import Iterable
import os

from .base import ClusterOutput
from ..io import PathConfig, Walk, walk_dataclass_jsonl


class ExtractionDirHandler:
    def __init__(self, paths: PathConfig):
        self.paths = paths

    def get_extraction_output_path(self, llm_path: str, walk: Walk, **kwargs) -> str:
        return self.make_output_path(self.paths.run_dir, llm_path, walk, **kwargs)

    @staticmethod
    def make_output_path(
        run_id_aware_root: str,
        llm_path: str,
        walk: Walk,
        **kwargs,
    ) -> str:
        return walk.map(os.path.join(run_id_aware_root, llm_path), **kwargs)

    @staticmethod
    def get_assign_id(walk: Walk) -> str:
        return walk.no_ext()

    def get_run_id_and_llm(self, walk: Walk) -> tuple[str, str]:
        relpath = os.path.relpath(walk.root, start=self.paths.extraction_dir)
        run_id = self._get_run_id(relpath)
        return run_id, os.path.relpath(relpath, start=run_id)

    @staticmethod
    def _get_run_id(path: str) -> str:
        top_level, current_level = None, path
        while True:
            current_level = os.path.dirname(current_level)
            if current_level in ["", "/"]:
                break
            else:
                top_level = current_level
        if top_level is None:
            raise ValueError
        return str(top_level)

    def walk_extractions(self) -> Iterable[tuple[list[ClusterOutput], Walk]]:
        yield from walk_dataclass_jsonl(self.paths.extraction_dir, t=ClusterOutput)
