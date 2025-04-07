from datetime import date
from typing import Union
from enum import Enum
import sys
import os

import coma
import pandas as pd

from .base import Configs as Cfgs, init

from ..analyze import HistogramMaker
from ..consolidate import ConsolidateConfig, Consolidator
from ..core import ClustersConfig, FewShotSampler
from ..extract import Extract, ExtractionDirHandler
from ..segmentation import ConvertTagsToTranscript, TagsConfig, Tagger, Transcript
from ..llms import (
    DummyConfig,
    LLMImplementation,
    LLMsConfig,
    MISSING_NICKNAME,
    TransformersConfig,
)
from ..io import (
    PathConfig,
    ensure_path,
    enum_from_str,
    walk_dataclass_json,
    walk_dataclass_jsonl,
    save_lines,
    save_json,
    save_dataclass_json,
    save_dataclass_jsonl,
    walk_docx,
    walk_files,
    walk_json,
)
from ..validation import (
    ComparisonAggregator,
    GroundTruthComparator,
    GroundTruthParser,
    ParsedLabelData,
    ValidationConfig,
    test_comparators,
)


def docx_to_json(paths: PathConfig):
    for lines, walk in walk_docx(paths.docx_transcript_dir):
        # Extract the 'assign ID' to create the output filename.
        a_id = walk.base.split("_")[0]
        output_file = str(os.path.join(paths.json_transcript_dir, a_id + ".json"))
        save_json(output_file, lines, indent=4)


def segment(paths: PathConfig, tags: TagsConfig, clusters: ClustersConfig):
    tag = Tagger(tags)
    to_transcript = ConvertTagsToTranscript(clusters)
    for lines, walk in walk_json(paths.json_transcript_dir):
        transcript = to_transcript(lines, tag(lines))
        output_file = walk.map(paths.clustered_transcript_dir)
        save_dataclass_json(output_file, transcript, indent=4)


class RerunProtocol(Enum):
    """The protocol for treating existing output files during a rerun."""

    NEVER = "NEVER"  # Never allow rerun. Raise an error if previous files exist.
    MISSING = "MISSING"  # Allow a partial rerun. Only run missing files.
    OVERWRITE = "OVERWRITE"  # Allow a full rerun. Overwrite every file.

    def skip(self, output_file: str):
        if not os.path.exists(output_file):
            return False
        if self == RerunProtocol.NEVER:
            raise ValueError(
                f"RerunProtocol set to '{self}' but file exists: {output_file}",
            )
        elif self == RerunProtocol.MISSING:
            return True
        elif self == RerunProtocol.OVERWRITE:
            return False
        else:
            raise ValueError(f"Unsupported RerunProtocol: {self}")


class ReferenceDate:
    def __init__(self, paths: PathConfig):
        df = pd.read_csv(paths.master_file, index_col=False)
        data = df[["assigned_id", "date_interview"]].to_dict()
        data = dict(zip(data["assigned_id"].values(), data["date_interview"].values()))
        self.data = {a_id: date.fromisoformat(d) for a_id, d in data.items()}

    def get(self, assign_id: str) -> date:
        return self.data[assign_id]


class ExtractCommand:
    def __init__(
        self,
        paths: PathConfig,
        clusters: ClustersConfig,
        llms: LLMsConfig,
        llm_impl_cfg: Union[DummyConfig, TransformersConfig],
        rerun_protocol: RerunProtocol,
    ):
        self.paths, self.clusters, self.llms = paths, clusters, llms
        self.rerun_protocol = rerun_protocol
        sampler = FewShotSampler(paths)
        self.do_extract = Extract(
            clusters=clusters,
            llms=llms,
            sampler=sampler,
            llm_cfg=llm_impl_cfg,
            dummy_cheat_sampler=sampler,
        )
        self.pathing = ExtractionDirHandler(paths)
        self.make_output_file = self.pathing.get_extraction_output_path
        self.get_reference = ReferenceDate(paths).get

    def run(self):
        root = self.paths.clustered_transcript_dir
        for transcript, walk in walk_dataclass_json(root, t=Transcript):
            output_file = self.make_output_file(self.llms.llm, walk, ext=".jsonl")
            if self.rerun_protocol.skip(output_file):
                continue
            a_id = self.pathing.get_assign_id(walk)
            extractions = self.do_extract(
                transcript,
                reference_date=self.get_reference(a_id),
                assign_id=a_id,
            )
            save_dataclass_jsonl(output_file, *extractions)


def parse_labels(paths: PathConfig, clusters: ClustersConfig):
    parser = GroundTruthParser(clusters)
    get_reference = ReferenceDate(paths).get
    for label_data, walk in walk_json(paths.labeled_transcript_dir):
        a_id = walk.no_ext()
        reference = get_reference(a_id)
        parsed_labels = parser(a_id, label_data, reference_date=reference)
        output_file = walk.map(paths.parsed_labels_dir, ext=".jsonl")
        save_dataclass_jsonl(output_file, *parsed_labels)


def mini_validation(
    paths: PathConfig,
    clusters: ClustersConfig,
    validation: ValidationConfig,
):
    labels_by_aid = {}
    pathing = ExtractionDirHandler(paths)
    labels_path = paths.parsed_labels_dir
    for parsed_labels, walk in walk_dataclass_jsonl(labels_path, t=ParsedLabelData):
        labels_by_aid[walk.no_ext()] = parsed_labels

    comparison_per_run_and_llm = {}
    compare = GroundTruthComparator(clusters)
    for data, walk in pathing.walk_extractions():
        a_id = pathing.get_assign_id(walk)
        run_id, llm = pathing.get_run_id_and_llm(walk)
        if run_id not in validation.run_ids_to_include:
            continue
        if llm not in validation.llms_to_include:
            continue
        comparisons = comparison_per_run_and_llm.setdefault((run_id, llm), [])
        comparisons.append(compare(a_id, data, labels_by_aid[a_id]))

    aggregate = ComparisonAggregator(validation.cluster_column_name)
    for (run_id, llm), comparisons in comparison_per_run_and_llm.items():
        result = aggregate(comparisons)
        output_dir = os.path.join(paths.aggregate_scores_dir, run_id, llm)
        for i, df in enumerate(result.few_shot):
            output_file = os.path.join(output_dir, f"few_shot_top_{i + 1}.csv")
            df.to_csv(ensure_path(output_file), index=False)
        for i, df in enumerate(result.other):
            output_file = os.path.join(output_dir, f"other_top_{i + 1}.csv")
            df.to_csv(ensure_path(output_file), index=False)


def consolidate_(
    paths: PathConfig,
    clusters: ClustersConfig,
    consolidate: ConsolidateConfig,
):
    pathing = ExtractionDirHandler(paths)
    Consolidator(consolidate, clusters, pathing)(paths.consolidate_dir)


def analyze_histogram(
    paths: PathConfig,
    clusters: ClustersConfig,
    consolidate: ConsolidateConfig,
    rerun_protocol: RerunProtocol,
):
    make_histograms = HistogramMaker(clusters, consolidate)
    for walk in walk_files(paths.consolidate_dir):
        output_file = walk.map(new_root=paths.analysis_dir, ext=".txt")
        if rerun_protocol.skip(output_file):
            continue
        save_lines(
            output_file,
            *make_histograms(walk.path),
            to_string_fn=lambda h: h.make_report() + "\n\n",
        )


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))
    coma.register("test.comp", test_comparators)

    coma.register("docx.to.json", docx_to_json, **Cfgs.add(Cfgs.paths))
    coma.register(
        "segment",
        segment,
        **Cfgs.add(Cfgs.paths, Cfgs.tags, Cfgs.clusters),
    )

    @coma.hooks.hook
    def extract_pre_config_hook(known_args, unknown_args, configs):
        """
        Preloads just the LLMsConfig to use its 'implementation' field to choose
        which LLM implementation's config to add to 'configs' (in place).
        """
        # Preload just the LLMsConfig.
        llm_cfg = coma.hooks.config_hook.single_load_and_write_factory(
            Cfgs.llms.id_,
            write_on_fnf=False,
        )(known_args=known_args, configs=configs)
        llm_cfg = coma.hooks.post_config_hook.single_cli_override_factory(
            Cfgs.llms.id_,
            coma.config.cli.override_factory(sep="::"),
        )(unknown_args=unknown_args, configs=llm_cfg)
        cfg: LLMsConfig = llm_cfg[Cfgs.llms.id_]

        # Add the right LLM config to 'configs' (in place).
        if cfg.llm == MISSING_NICKNAME:
            raise ValueError(f"Missing runtime LLM: {cfg.llm}")
        if cfg.implementation == LLMImplementation.MISSING:
            raise ValueError(f"Missing implementation type for LLM: {cfg.llm}")
        elif cfg.implementation == LLMImplementation.DUMMY:
            llm_impl = Cfgs.dummy
        elif cfg.implementation == LLMImplementation.HF_TRANSFORMERS:
            llm_impl = Cfgs.transformers
        else:
            raise ValueError(
                f"Unsupported LLM implementation type: {cfg.implementation}",
            )
        configs[llm_impl.id_] = llm_impl.type_

    rerun_parser_hook = coma.hooks.parser_hook.factory(
        "-p",
        "--rerun-protocol",
        default="never",
        choices=["never", "missing", "overwrite"],
        help="set the protocol for treating existing output files during rerun",
    )

    @coma.hooks.hook
    def rerun_pre_init_hook(known_args, configs):
        protocol = enum_from_str(RerunProtocol, known_args.rerun_protocol)
        configs["rerun_protocol"] = protocol

    coma.register(
        "extract",
        ExtractCommand,
        parser_hook=rerun_parser_hook,
        pre_config_hook=extract_pre_config_hook,
        pre_init_hook=rerun_pre_init_hook,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.llms),
    )

    coma.register("parse.labels", parse_labels, **Cfgs.add(Cfgs.paths, Cfgs.clusters))
    coma.register(
        "mini.validation",
        mini_validation,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.validation),
    )
    coma.register(
        "consolidate",
        consolidate_,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.consolidate),
    )
    coma.register(
        "analyze.histogram",
        analyze_histogram,
        parser_hook=rerun_parser_hook,
        pre_init_hook=rerun_pre_init_hook,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.consolidate),
    )


def launch():
    """Launches the application with Coma."""
    init()
    register()
    try:
        coma.wake()
    except AttributeError:
        if len(sys.argv) == 1:
            os.chdir(os.environ["DEFAULT_CONFIG_DIR"])
            coma.wake(args=[os.environ["DEFAULT_COMMAND"]])
        else:
            raise
