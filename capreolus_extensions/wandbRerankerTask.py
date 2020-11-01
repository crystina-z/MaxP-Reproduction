import shutil
from pathlib import Path
from collections import defaultdict
from profane import ConfigOption

from capreolus import evaluator
from capreolus.searcher import Searcher
from capreolus.sampler import PredSampler
from capreolus.task import Task, RerankTask
from capreolus.utils.loginit import get_logger

from utils import get_wandb

wandb = get_wandb()
logger = get_logger(__name__)


@Task.register
class WandbRerankerTask(RerankTask):
    module_name = "wandbreranker"
    config_spec = RerankTask.config_spec  # ["fold", "optimize", "threshold"]
    metrics = evaluator.DEFAULT_METRICS

    def train(self, init_path=""):
        fold = self.config["fold"]

        self.rank.search()
        rank_results = self.rank.evaluate()
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)
        wandb.save(str(best_search_run_path))
        return self.rerank_run(best_search_run, self.get_results_path(), init_path=init_path)

    def rerank_run(self, best_search_run, train_output_path, include_train=False, init_path=""):
        if not isinstance(train_output_path, Path):
            train_output_path = Path(train_output_path)

        fold = self.config["fold"]
        threshold = self.config["threshold"]
        dev_output_path = train_output_path / "pred" / "dev"
        logger.debug("results path: %s", train_output_path)

        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
        self.reranker.extractor.preprocess(
            qids=best_search_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type])

        self.reranker.build_model()
        self.reranker.searcher_scores = best_search_run

        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}

        # For each qid, select the top 100 (defined by config["threshold") docs to be used in validation
        dev_run = defaultdict(dict)
        # This is possible because best_search_run is an OrderedDict
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        assert len(dev_run[qid]) == threshold, f"Expect {threshold} on each qid, got {len(dev_run[qid])} for query {qid}"
                        break
                    dev_run[qid][docid] = score

        # Depending on the sampler chosen, the dataset may generate triplets or pairs
        train_dataset = self.sampler
        train_dataset.prepare(
            train_run, self.benchmark.qrels, self.reranker.extractor,
            relevance_level=self.benchmark.relevance_level,
        )
        dev_dataset = PredSampler()
        dev_dataset.prepare(
            dev_run, self.benchmark.qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level,
        )

        train_args = [self.reranker, train_dataset, train_output_path, dev_dataset, dev_output_path, self.benchmark.qrels, self.config["optimize"], self.benchmark.relevance_level]
        if self.reranker.trainer.module_name == "tensorflowlog":
            self.reranker.trainer.train(*train_args, init_path=init_path)
        else:
            self.reranker.trainer.train(*train_args)

        self.reranker.trainer.load_best_model(self.reranker, train_output_path)
        dev_output_path = train_output_path / "pred" / "dev" / "best"
        dev_preds = self.reranker.trainer.predict(self.reranker, dev_dataset, dev_output_path)
        shutil.copy(dev_output_path, dev_output_path.parent / "dev.best")
        wandb.save(str(dev_output_path.parent / "dev.best"))

        test_run = defaultdict(dict)
        # This is possible because best_search_run is an OrderedDict
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["test"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        assert len(test_run[qid]) == threshold, f"Expect {threshold} on each qid, got {len(dev_run[qid])} for query {qid}"
                        break
                    test_run[qid][docid] = score

        test_dataset = PredSampler()
        test_dataset.prepare(
            test_run, self.benchmark.unsampled_qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level)
        test_output_path = train_output_path / "pred" / "test" / "best"
        test_preds = self.reranker.trainer.predict(self.reranker, test_dataset, test_output_path)
        shutil.copy(test_output_path, test_output_path.parent / "test.best")
        wandb.save(str(test_output_path.parent / "test.best"))

        preds = {"dev": dev_preds, "test": test_preds}

        if include_train:
            train_dataset = PredSampler(
                train_run, self.benchmark.qrels, self.reranker.extractor,
                relevance_level=self.benchmark.relevance_level,
            )

            train_output_path = train_output_path / "pred" / "train" / "best"
            train_preds = self.reranker.trainer.predict(self.reranker, train_dataset, train_output_path)
            preds["train"] = train_preds

        return preds

    def predict_and_eval(self, init_path=None):
        fold = self.config["fold"]
        self.reranker.build_model()
        if not init_path or init_path == "none":
            logger.info(f"Loading self best ckpt: {init_path}")
            logger.info("No init path given, using default parameters")
            self.reranker.build_model()
        else:
            logger.info(f"Load from {init_path}")
            init_path = Path(init_path) if not init_path.startswith("gs:") else init_path
            self.reranker.trainer.load_best_model(self.reranker, init_path, do_not_hash=True)

        dirname = str(init_path).split("/")[-1] if init_path else "noinitpath"
        savedir = Path(__file__).parent.absolute() / "downloaded_runfiles" / dirname
        dev_output_path = savedir / fold / "dev"
        test_output_path = savedir / fold / "test"
        test_output_path.parent.mkdir(exist_ok=True, parents=True)

        self.rank.search()
        threshold = self.config["threshold"]
        rank_results = self.rank.evaluate()
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
        self.reranker.extractor.preprocess(
            qids=best_search_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type])

        # dev run
        dev_run = defaultdict(dict)
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        assert len(dev_run[qid]) == threshold, f"Expect {threshold} on each qid, got {len(dev_run[qid])} for query {qid}"
                        break
                    dev_run[qid][docid] = score
        dev_dataset = PredSampler()
        dev_dataset.prepare(
            dev_run, self.benchmark.qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level)

        # test_run
        test_run = defaultdict(dict)
        # This is possible because best_search_run is an OrderedDict
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["test"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        assert len(test_run[qid]) == threshold, f"Expect {threshold} on each qid, got {len(dev_run[qid])} for query {qid}"
                        break
                    test_run[qid][docid] = score

        unsampled_qrels = self.benchmark.unsampled_qrels if hasattr(self.benchmark, "unsampled_qrels") else self.benchmark.qrels
        test_dataset = PredSampler()
        test_dataset.prepare(
            test_run, unsampled_qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level)
        logger.info("test prepared")

        # prediction
        dev_preds = self.reranker.trainer.predict(self.reranker, dev_dataset, dev_output_path)
        fold_dev_metrics = evaluator.eval_runs(
            dev_preds, unsampled_qrels, self.metrics, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s dev metrics: %s", fold, fold_dev_metrics)

        test_preds = self.reranker.trainer.predict(self.reranker, test_dataset, test_output_path)
        fold_test_metrics = evaluator.eval_runs(
            test_preds, unsampled_qrels, self.metrics, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s test metrics: %s", fold, fold_test_metrics)
        wandb.save(str(dev_output_path))
        wandb.save(str(test_output_path))

        # add cross validate results:
        n_folds = len(self.benchmark.folds)
        folds_fn = {f"s{i}": savedir / f"s{i}" / "test" for i in range(1, n_folds+1)}
        if not all([fn.exists() for fn in folds_fn.values()]):
            return {"fold_test_metrics": fold_test_metrics, "cv_metrics": None}

        all_preds = {}
        reranker_runs = {
            fold: {
                "dev": Searcher.load_trec_run(fn.parent / "dev"),
                "test": Searcher.load_trec_run(fn)}
            for fold, fn in folds_fn.items()}

        for fold, dev_test in reranker_runs.items():
            preds = dev_test["test"]
            qids = self.benchmark.folds[fold]["predict"]["test"]
            for qid, docscores in preds.items():
                if qid not in qids:
                    continue
                all_preds.setdefault(qid, {})
                for docid, score in docscores.items():
                    all_preds[qid][docid] = score

        cv_metrics = evaluator.eval_runs(
            all_preds, unsampled_qrels, self.metrics, self.benchmark.relevance_level
        )
        for metric, score in sorted(cv_metrics.items()):
            logger.info("%25s: %0.4f", metric, score)

        searcher_runs = {}
        rank_results = self.rank.evaluate()
        for fold in self.benchmark.folds:
            searcher_runs[fold] = {"dev": Searcher.load_trec_run(rank_results["path"][fold])}
            searcher_runs[fold]["test"] = searcher_runs[fold]["dev"]

        interpolated_results = evaluator.interpolated_eval(
            searcher_runs, reranker_runs, self.benchmark, self.config["optimize"], self.metrics
        )

        return {
            "fold_test_metrics": fold_test_metrics,
            "cv_metrics": cv_metrics,
            "interpolated_results": interpolated_results,
        }

    def evaluate(self):
        fold = self.config["fold"]
        train_output_path = self.get_results_path()
        logger.debug("results path: %s", train_output_path)

        searcher_runs, reranker_runs = self.find_crossvalidated_results()

        if fold not in reranker_runs:
            logger.error("could not find predictions; run the train command first")
            raise ValueError("could not find predictions; run the train command first")

        fold_dev_metrics = evaluator.eval_runs(
            reranker_runs[fold]["dev"], self.benchmark.qrels, self.metrics, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s dev metrics: %s", fold, fold_dev_metrics)

        unsampled_qrels = self.benchmark.unsampled_qrels if hasattr(self.benchmark, "unsampled_qrels") else self.benchmark.qrels
        fold_test_metrics = evaluator.eval_runs(
            reranker_runs[fold]["test"], unsampled_qrels, self.metrics, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s test metrics: %s", fold, fold_test_metrics)

        if len(reranker_runs) != len(self.benchmark.folds):
            logger.info(
                "rerank: skipping cross-validated metrics because results exist for only %s/%s folds",
                len(reranker_runs),
                len(self.benchmark.folds))
            logger.info("available runs: ", reranker_runs.keys())
            return {
                "fold_test_metrics": fold_test_metrics,
                "fold_dev_metrics": fold_dev_metrics,
                "cv_metrics": None,
                "interpolated_cv_metrics": None,
            }

        logger.info("rerank: average cross-validated metrics when choosing iteration based on '%s':", self.config["optimize"])
        all_preds = {}
        for preds in reranker_runs.values():
            for qid, docscores in preds["test"].items():
                all_preds.setdefault(qid, {})
                for docid, score in docscores.items():
                    all_preds[qid][docid] = score

        cv_metrics = evaluator.eval_runs(
            all_preds, unsampled_qrels, self.metrics, self.benchmark.relevance_level
        )
        interpolated_results = evaluator.interpolated_eval(
            searcher_runs, reranker_runs, self.benchmark, self.config["optimize"], self.metrics
        )

        for metric, score in sorted(cv_metrics.items()):
            logger.info("%25s: %0.4f", metric, score)

        return {
            "fold_test_metrics": fold_test_metrics,
            "fold_dev_metrics": fold_dev_metrics,
            "cv_metrics": cv_metrics,
            "interpolated_results": interpolated_results,
        }
