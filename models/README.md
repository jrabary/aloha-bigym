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


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.


to eval: 


# number 1

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0 --eval

# number 2
mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 5e-5 \
    --seed 0 --eval

# number 3

mjpython model_eval.py \
    --task_name sim_aloha_close_dishwasher \
    --ckpt_dir ./ckpt_dir \
    --policy_class ACT --kl_weight 100 --chunk_size 3 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 5e-5 \
    --seed 0 --eval