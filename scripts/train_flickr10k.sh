python train.py --multiprocessing_distributed \
    --train_data_path /path/to/Flickr-all/ \
    --test_data_path /path/to/Flickr-SoundNet/ \
    --test_gt_path /path/to/Flickr-SoundNet/Annotations/ \
    --experiment_name flickr_10k \
    --trainset 'flickr_10k' \
    --testset 'flickr' \
    --epochs 100 \
    --batch_size 128 \
    --init_lr 0.0001