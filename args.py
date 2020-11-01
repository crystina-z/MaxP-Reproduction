from argparse import ArgumentParser

import numpy as np


TASKS = [
    "optimal",  # to reproduce the optimimal result of the model
    "sampling",  # data sampling
    "inference",  # inference from given ckpt
    "bertology",  # train maxp with different pretrained models
]


def _float2str(x):
    return f"%.2f" % x


def get_args(*args):
    def string2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        raise ValueError(f"Unexpected value: {v}")

    parser = ArgumentParser()

    parser.add_argument("--project_name", default="ecir2021_reproduce")
    parser.add_argument("--task", default="optimal", choices=TASKS)
    parser.add_argument("--model", default="maxp", choices=["maxp"])  # more models in the future
    parser.add_argument("--dataset", default="rob04", choices=["rob04", "gov2"])
    parser.add_argument("--gov2_path", type=str)  # only needed to GOV2 collection

    parser.add_argument("--use_prepared_gov2_runfile", type=string2bool, default=True)

    parser.add_argument("--fold", type=str, default="s1")
    parser.add_argument("--query_type", type=str, default="title")
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "first", "sum"])

    parser.add_argument("--init_path", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased")

    parser.add_argument("--use_cache", type=str, default="True")
    parser.add_argument("--train", type=string2bool, default=True)
    parser.add_argument("--eval", type=string2bool, default=True)

    parser.add_argument("--sampling_rate", "-rate", type=float, default=1.0)
    parser.add_argument("--nhits", type=str, default="1000")

    # tpu settings
    parser.add_argument("--tpu", type=str, default="None")
    parser.add_argument("--gs_storage", type=str, default="None")
    parser.add_argument("--tpuzone", type=str, default="None")

    return parser.parse_args(*args)


def get_task_config(args):
    task = args.task
    if task in ["optimal", "inference"]:
        return []

    if task == "sampling":
        rates = [args.sampling_rate] \
            if args.sampling_rate != "all" else list(map(_float2str, np.arange(0.1, 1, 0.2))) + [1.0]
        return [{"benchmark.rate": rate} for rate in rates]

    if task == "bertology":
        pretrained = args.pretrained
        pretrained_tokenizer = pretrained
        if "electra" in pretrained or "bert-large-uncased" in pretrained:
            pretrained_tokenizer = "bert-base-uncased"  # to reuse the cache as much as possible

        return [{
            "reranker.extractor.tokenizer.pretrained": pretrained_tokenizer,
            "reranker.pretrained": pretrained,
        }]

    raise ValueError(f"Unrecognized task type: {task}")

