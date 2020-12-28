import os
import shutil
from pathlib import Path

from capreolus import ConfigOption, Dependency, constants, parse_config_string
from capreolus.index import Index, AnseriniIndex
from capreolus.searcher import Searcher
from capreolus.searcher.anserini import BM25


@Index.register
class GovIndex(AnseriniIndex):
    module_name = "gov2index"
    path = ""  # store the anserini index
    dependencies = [
        Dependency(key="collection", module="collection", name="gov2collection")
    ]

    def get_index_path(self):
        return self.path

    def exists(self):
        return True

    def get_doc(self, doc_id):
        return self.collection.get_doc(doc_id)


@Searcher.register
class GovSearcher(BM25):
    module_name = "gov2searcher"
    dependencies = [
        Dependency(key="index", module="index", name="gov2index")
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        done_fn = output_path / "done"
        if done_fn.exists():
            return output_path

        assert config["k1"] == (0.9,) 
        assert config["b"] == (0.4,)
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True) 
        src = Path(__file__).parent.parent / "data" / "gov2_bm25" / "run_k1=0.9,b=0.4"
        dest = output_path / "searcher"
        shutil.copyfile(src.as_posix(), dest.as_posix())

        with open(done_fn, "w") as f:
            f.write("done")
        return output_path
