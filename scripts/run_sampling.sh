fold=s1       # s1-s5 for Robust04, s1 to s3 for GOV2
dataset=rob04 # supports rob04 or gov2,
              # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below
do_train=True # if False, training will be skipped. Acceptable if training results are already available
do_eval=True  # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True"

rate=1.0      # 0.10 0.30 0.50 0.70 0.90 1.0

python run.py \
    --task sampling \
    --dataset $dataset \
    --sampling_rate $rate \
    --fold $fold \
    --train $do_train \
    --eval $do_eval

    # to use TPU, add:
    # --tpu your_tpu_name --tpuzone your_tpu_zone (e.g. us-central1-f) --gs_bucket gs://your_gs_bucket_path

    # to use WandB, add
    # --project_name your_wandb_project_name`
