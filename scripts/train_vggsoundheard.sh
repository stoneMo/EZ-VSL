python train.py --multiprocessing_distributed \
    --train_data_path /path/to/VGGSound-all/ \
    --test_data_path /path/to/VGGSound-test/ \
    --experiment_name vggss_heard \
    --trainset 'vggss_heard' \
    --testset 'vggss_heard' \
    --epochs 20 \
    --batch_size 128 \
    --init_lr 0.0001