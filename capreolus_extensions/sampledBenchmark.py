import json
from pathlib import Path

from capreolus import ConfigOption, Dependency, constants, parse_config_string
from capreolus.benchmark import Benchmark
from capreolus.benchmark.robust04 import Robust04Yang19
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels, topic_to_trectxt

PACKAGE_PATH = constants["PACKAGE_PATH"]
logger = get_logger(__name__)

DATA_PATH = (Path(__file__).parent.parent / "data").absolute()


@Benchmark.register
class GOV2Benchmark(Benchmark):
    module_name = "gov2benchmark"
    query_type = "title"

    qrel_file = DATA_PATH / module_name / "qrels.gov2.txt"
    topic_file = DATA_PATH / module_name / "topics.gov2.701-750.751-800.801-850.txt"
    fold_file = DATA_PATH / module_name / "gov2.json"
    dependencies = [Dependency(key="collection", module="collection", name="gov2collection")]


class SampleMixin:
    @property
    def unsampled_qrels(self):
        if not hasattr(self, "_unsampled_qrels"):
            self._unsampled_qrels = load_qrels(self.unsampled_qrel_file)
        return self._unsampled_qrels

    @property
    def unsampled_folds(self):
        if not hasattr(self, "_unsampled_folds"):
            self._unsampled_folds = json.load(open(self.unsampled_fold_file, "rt"))
        return self._unsampled_folds

    def build(self):
        rate = self.config["rate"]
        self.unsampled_qrel_file = self.qrel_file
        self.unsampled_fold_file = self.fold_file
        if rate < 1:  # else still use the original qrel_file and fold_file as the "sampled" one 
            self.qrel_file = self.file_fn / ("rate=%.2f.qrels.txt" % rate)
            self.fold_file = self.file_fn / ("rate=%.2f.fold.json" % rate)
        assert all([f.exists() for f in [
            self.unsampled_qrel_file, self.unsampled_fold_file, self.qrel_file, self.fold_file]])


@Benchmark.register
class SampledRobust04(Robust04Yang19, SampleMixin):
    module_name = "sampled_rob04.title"
    file_fn = DATA_PATH / module_name
    config_spec = [ConfigOption("rate", 1.0, "sampling rate: fraction number between 0 to 1")]


@Benchmark.register
class SampledGOV2(GOV2Benchmark, SampleMixin):
    module_name = "sampled_gov2.title"
    file_fn = DATA_PATH / module_name
    config_spec = [ConfigOption("rate", 1.0, "sampling rate: fraction number between 0 to 1")]


@Benchmark.register
class SampledRobust04Desc(SampledRobust04):
    module_name = "sampled_rob04.desc"
    query_type = "desc"


@Benchmark.register
class SampledGOV2Desc(SampledGOV2):
    module_name = "sampled_gov2.desc"
    query_type = "desc"
