## Comparing Score Aggregation Approaches for Pretrained Neural Language Models

This paper contains the code to reproduce the results in the 
ECIR2020 reproducibility paper: *Comparing Score Aggregation Approaches for Pretrained Neural Language Models.*
Note that since we highly rely on the *ad-hoc retrieval framework [capreolus](https://capreolus.ai/)*, 
the modules in this repo are mostly the extension of the framework (under `./capreolus_extensions`) and do not contain the data processing and training logic.
Please find the details in the [framework github](https://github.com/capreolus-ir/capreolus) if you are interested in these.

The hyperparameters are listed under `./optimal_configs/maxp.txt`, with the format `config_key=config_value` each line.
Feel free to try other settings using this format. Note that lines starting with `#` is considered as comments and will be ignored by the program.
For the config key format and acceptable values, please find more details [here](https://capreolus.ai/en/latest/quick.html#command-line-interface). 

### Setup
```
# install capreolus
pip install git+https://github.com/capreolus-ir/capreolus.git@feature/autotokenizer
pip uninstall h5py && pip install h5py==2.10.0 

# designate the directory under which the cache and results will be stored 
CAPREOLUS_CACHE=/on/spacious/disk/.capreolus/cache
CAPREOLUS_RESULTS=/on/spacious/disk/.capreolus/results
```

#### (For WandB users)
If you use wandb, simply install wandb and login in the standard way:
```
pip install wandb
wandb login
``` 
Then the results can be easily synced to your project by simply adding `--project_name your_wandb_project_name`. 
You are expected to see all configs and the value of metric `mAP`, `P@20`, `nDCG@20` plotted. 

### Replicate paper results 
The code is written in tensorflow-2.3 and supports GPU and TPU v2/v3 (by Capreolus). 
This section provides the code to replicate all the experiments listed in the paper, 
which can be also found under `./scripts`

#### Section 1: Reproduce FirstP, MaxP, SumP (and AvgP)
The following script is used to **train** the experiments with sampled dataset. 
```
fold=s1         # s1-s5 for robust04, s1-s3 for gov2
dataset=rob04   # supports rob04 or gov2,
                # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below
do_train=True   # if False, training will be skipped. Acceptable if training results are already available
do_eval=True    # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True"

query_type=title    # title or desc
aggregation=max     # max, avg, first or sum

python run.py \
  --task bertology \
  --dataset $dataset \
  --query_type $query_type \
  --fold $fold \
  --aggregation $aggregation \
  --train $do_train --eval $do_eval
```

When all folds results are available, the program will also show cross-validated results on the evaluation stage. 

#### Section 2: MaxP with BERT variants
```
fold=s1         # s1-s5 for robust04, s1-s3 for gov2
dataset=rob04   # supports rob04 or gov2,
                # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below
do_train=True   # if False, training will be skipped. Acceptable if training results are already available
do_eval=True    # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True"

query_type=title    # title or desc
pretrained=bert-base-uncased # one of electra-base-msmarco google/electra-base-discriminator bert-base-uncased albert-base-v2 roberta-base bert-large-uncased
aggregation=max     # max, avg, first or sum

python run.py \
  --task bertology \
  --dataset $dataset \
  --pretrained $pretrained \
  --query_type $query_type \
  --fold $fold \
  --train $do_train --eval $do_eval
```

#### Section 3: MaxP with Sampled Dataset 
```
fold=s1       # s1-s5 for Robust04, s1 to s3 for GOV2
dataset=rob04 # supports rob04 or gov2,
              # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below
do_train=True # if False, training will be skipped. Acceptable if training results are already available
do_eval=True  # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True"

rate=1.0      # supports (0.0, 1.0], where 1.0 (default) means no sampling will be done

python run.py \
    --task sampling \
    --dataset $dataset \
    --sampling_rate $rate \
    --fold $fold \
    --train $do_train --eval $do_eval 
```


#### TPU users
If TPU is available, append the following arguments to the above scripts to run the experiments on TPU: 
```
--tpu your_tpu_name --tpuzone your_tpu_zone (e.g. us-central1-f) --gs_bucket gs://your_gs_bucket_path
``` 


#### GOV2 
As GOV2 is not public dataset, users need to prepare the dataset to test on GOV2 dataset.
Once the collection is prepared, specify the GOV2 directory through `--gov2_path /path/to/GOV2`,
where `ls /path/to/GOV2` should present a list of subdirectories from `GX000` to `GX272`. 
