The code in this folder is directly from Tony Zhao's ACT: [https://github.com/tonyzhaozh/act/tree/main](https://github.com/tonyzhaozh/act/tree/main)
Modified to work with joycon virtual teleop setup




to train run


To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0

### number 3

!cd /content/act-bigym-aloha && mjpython imitate_episodes.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 100 --chunk_size 3 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 \
    --num_epochs 500  --lr 3e-4 \
    --seed 0

### number 4

!cd /content/act-bigym-aloha && python3 imitate_episodes.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 5e-5 \
    --seed 1

### number 5

!cd /content/act-bigym-aloha && python3 imitate_episodes.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 50 --chunk_size 15 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 5e-5 \
    --seed 1 



# EVAL

To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.


to eval: 


### number 1

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0 --eval

### number 2
mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 5e-5 \
    --seed 0 --eval

### number 3

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 100 --chunk_size 3 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 \
    --num_epochs 500  --lr 3e-4 \
    --seed 0 --eval


### number 4
need to increase the chunk size cause 3 in this sim is just too low
also lowkey i could increase the data gathering frequency, can do that if this training run is mid

probs go back to previous batch size and lr as well

num epochs should be 3000 tbh

i think previous kl_weight is good, want to mirror the overall dist but not mirror the odd movements

hmm bu tcurrent is sort of off
see how exactly the kl_weight varies the prioritiies

decided kl weight 50

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 50 --chunk_size 10 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 5e-5 \
    --seed 1 --eval


### number 5

i think everything good except i want to increase chunk size more, the having to wait during inference is p annoying

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 50 --chunk_size 15 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 5e-5 \
    --seed 1 --eval

### number 6

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 50 --chunk_size 15 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 5e-5 \
    --seed 4 --eval


