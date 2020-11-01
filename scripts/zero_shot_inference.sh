ckpt_path=/path/to/ckpt  # Note that for TPU users, the ckpt needs to be uploaded to gcs
                         # Then this path would look like gs://path/to/ckpt
fold=s1
dataset=rob04
other_command=$@

python run.py --task inference --dataset rob04 --fold $fold $other_command

# to use TPU, add:
# --tpu your_tpu_name --tpuzone your_tpu_zone (e.g. us-central1-f) --gs_bucket gs://your_gs_bucket_path

# to use WandB, add
# --project_name your_wandb_project_name`
