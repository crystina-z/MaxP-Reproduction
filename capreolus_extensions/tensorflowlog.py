import os
import hashlib
from tqdm import tqdm
from profane import ConfigOption

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from capreolus import evaluator
from capreolus.trainer import Trainer, TensorflowTrainer
from capreolus.utils.loginit import get_logger

from utils import get_wandb

wandb = get_wandb()
logger = get_logger(__name__)


def revise_ckpt_path(tvars, ckpt_fn):
    cur_names = [v.name for v in tvars]

    if "_1" in cur_names[0]:
        ckpt_fn = ckpt_fn.replace("/converted_on_the_run/", "/converted_on_the_run_with1/")
    else:
        ckpt_fn = ckpt_fn.replace("/converted_on_the_run_with1/", "/converted_on_the_run/")

    print([n for n in cur_names if "classifier" in n])
    ckpt_var = tf.train.list_variables(ckpt_fn)
    print([n for n, s in ckpt_var if "classifier" in n])
    return ckpt_fn


@tf.function
def check_same(ckpt_fn, tvars):
    var_names, ckpt_reader = tf.train.list_variables(ckpt_fn), tf.train.load_checkpoint(ckpt_fn)
    tvars_dict = {v.name: v for v in tvars}
    previous_tvars = {name: tf.convert_to_tensor(v) for name, v in tvars_dict.items()}
    for name, shape in var_names:
        if name not in tvars_dict:
            # print("Warning: ", name, shape, "not found", end="\t")
            name_0 = f"{name}:0"
            if name_0 in tvars_dict:
                # print(f"YET CAN BE FOUND")
                pass
            else:
                # print()
                continue
        else:
            name_0 = name

        a = tf.reduce_sum(tf.math.abs(ckpt_reader.get_tensor(name) - tvars_dict[name_0])),
        b = tf.reduce_sum(tf.math.abs(tvars_dict[name_0] - previous_tvars[name_0]))
        tf.print("\t\t >>>", name, a, b)
    return a + b


@Trainer.register
class TensorflowLogTrainer(TensorflowTrainer):
    module_name = "tensorflowlog"
    config_spec = TensorflowTrainer.config_spec + [
        ConfigOption("warmupbert", True, "whether to apply warmup on bert variables"),
        ConfigOption("warmupnonbert", True, "whether to apply warmup on nonbert variables"),
    ]

    def change_lr(self, epoch, lr, do_warmup):
        """ Apply warm up or decay depending on the current epoch """
        warmup_steps = self.config["warmupsteps"]
        if warmup_steps and epoch <= warmup_steps:
            return min(lr * ((epoch + 1) / warmup_steps), lr) if do_warmup else lr
        elif self.config["decaytype"] == "exponential":
            return lr * self.config["decay"] ** ((epoch - warmup_steps) / self.config["decaystep"])
        elif self.config["decaytype"] == "linear":
            return lr * (1 / (1 + self.config["decay"] * epoch))

        return lr

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path,
              qrels, metric, relevance_level=1, init_path=None):
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )
        os.makedirs(dev_output_path, exist_ok=True)
        start_epoch = self.config["niters"] if reranker.config.get("modeltype", "") in ["nir", "cedr"] else 0
        train_records = self.get_tf_train_records(reranker, train_dataset)
        dev_records = self.get_tf_dev_records(reranker, dev_data)
        dev_dist_dataset = self.strategy.experimental_distribute_dataset(dev_records)

        # Does not very much from https://www.tensorflow.org/tutorials/distribute/custom_training
        strategy_scope = self.strategy.scope()
        with strategy_scope:
            reranker.build_model()
            wrapped_model = self.get_wrapped_model(reranker.model)
            if init_path:
                logger.info(f"Initializing model from checkpoint {init_path}")
                print("number of vars: ", len(wrapped_model.trainable_variables))
                wrapped_model.load_weights(init_path)

            loss_object = self.get_loss(self.config["loss"])
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=self.config["lr"])
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate=self.config["bertlr"])

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config["batch"])

            def is_bert_parameters(name):
                name = name.lower()
                '''
                if "layer" in name:
                    if not ("9" in name or "10" in name or "11" in name or "12" in name):
                        return False
                '''
                if "/bert/" in name:
                    return True
                if "/electra/" in name:
                    return True
                if "/roberta/" in name:
                    return True
                if "/albert/" in name:
                    return True
                return False

        def train_step(inputs):
            data, labels = inputs

            with tf.GradientTape() as tape:
                train_predictions = wrapped_model(data, training=True)
                loss = compute_loss(labels, train_predictions)

            gradients = tape.gradient(loss, wrapped_model.trainable_variables)

            # TODO: Expose the layer names to lookout for as a ConfigOption?
            # TODO: Crystina mentioned that hugging face models have 'bert' in all the layers (including classifiers). Handle this case
            bert_variables = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if is_bert_parameters(variable.name) and "classifier" not in variable.name
            ]
            classifier_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if "classifier" in variable.name
            ]
            other_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if (not is_bert_parameters(variable.name)) and "classifier" not in variable.name
            ]

            assert len(bert_variables) + len(classifier_vars) + len(other_vars) == len(
                wrapped_model.trainable_variables)
            # TODO: Clean this up for general use
            # Making sure that we did not miss any variables

            if self.config["lr"] > 0:
                optimizer_1.apply_gradients(classifier_vars + other_vars)
            if self.config["bertlr"] > 0:
                optimizer_2.apply_gradients(bert_variables)

            return loss

        def test_step(inputs):
            data, labels = inputs
            predictions = wrapped_model.predict_step(data)

            return predictions

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = self.strategy.run(train_step, args=(dataset_inputs,))

            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return self.strategy.run(test_step, args=(dataset_inputs,))

        best_metric = -np.inf
        epoch = 0
        num_batches = 0
        total_loss = 0
        iter_bar = tqdm(total=self.config["itersize"])

        initial_lr = self.change_lr(epoch, self.config["bertlr"], do_warmup=self.config["warmupbert"])
        K.set_value(optimizer_2.lr, K.get_value(initial_lr))
        wandb.log({"bertlr": K.get_value(initial_lr)}, step=epoch + start_epoch, commit=False)

        initial_lr = self.change_lr(epoch, self.config["lr"], do_warmup=self.config["warmupnonbert"])
        K.set_value(optimizer_1.lr, K.get_value(initial_lr))
        wandb.log({"lr": K.get_value(initial_lr)}, step=epoch + start_epoch, commit=False)

        train_records = train_records.shuffle(100000)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_records)

        # Goes through the dataset ONCE (i.e niters * itersize * batch samples). However, the dataset may already contain multiple instances of the same sample,
        # depending upon what Sampler was used. If you want multiple epochs, achieve it by tweaking the niters and
        # itersize values.
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            train_loss = total_loss / num_batches
            num_batches += 1
            iter_bar.update(1)

            if num_batches % self.config["itersize"] == 0:
                epoch += 1

                # Do warmup and decay
                new_lr = self.change_lr(epoch, self.config["bertlr"], do_warmup=self.config["warmupbert"])
                K.set_value(optimizer_2.lr, K.get_value(new_lr))
                wandb.log({f"bertlr": K.get_value(new_lr)}, step=epoch + start_epoch, commit=False)

                new_lr = self.change_lr(epoch, self.config["lr"], do_warmup=self.config["warmupnonbert"])
                K.set_value(optimizer_1.lr, K.get_value(new_lr))
                wandb.log({f"lr": K.get_value(new_lr)}, step=epoch + start_epoch, commit=False)

                iter_bar.close()
                logger.info("train_loss for epoch {} is {}".format(epoch, train_loss))
                wandb.log({f"loss": float(train_loss.numpy())}, step=epoch + start_epoch, commit=False)
                total_loss = 0

                if epoch % self.config["validatefreq"] == 0:
                    dev_predictions = []
                    for x in tqdm(dev_dist_dataset, desc="validation"):
                        pred_batch = (
                            distributed_test_step(x).values
                            if self.strategy.num_replicas_in_sync > 1
                            else [distributed_test_step(x)]
                        )
                        for p in pred_batch:
                            dev_predictions.extend(p)

                    trec_preds = self.get_preds_in_trec_format(dev_predictions, dev_data)
                    metrics = evaluator.eval_runs(
                        trec_preds, dict(qrels),
                        evaluator.DEFAULT_METRICS + ["bpref"],
                        relevance_level)
                    logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    if metrics[metric] > best_metric:
                        logger.info("Writing checkpoint")
                        best_metric = metrics[metric]
                        wrapped_model.save_weights("{0}/dev.best".format(train_output_path))

                    wandb.log({
                        f"dev-{k}": v for k, v in metrics.items() if
                        k in ["map", "bpref", "P_20", "ndcg_cut_20", "judged_10", "judged_20", "judged_200"]},
                        step=epoch + start_epoch, commit=False)

                iter_bar = tqdm(total=self.config["itersize"])

            if num_batches >= self.config["niters"] * self.config["itersize"]:
                break

    def get_best_model_path(self, train_output_path):
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest())
        return "{0}/dev.best".format(train_output_path)

    def load_best_model(self, reranker, train_output_path, do_not_hash=False):
        # TODO: Do the train_output_path modification at one place?
        if self.tpu and not do_not_hash:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        reranker.build_model()
        # Because the saved weights are that of a wrapped model.
        wrapped_model = self.get_wrapped_model(reranker.model)
        best_path = "{0}/dev.best".format(train_output_path)
        print(best_path)
        wrapped_model.load_weights("{0}/dev.best".format(train_output_path))

        logger.info(f"Weights loaded from path {train_output_path}")
        return wrapped_model.model
