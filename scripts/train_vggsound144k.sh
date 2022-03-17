python train.py --multiprocessing_distributed \
    --train_data_path /path/to/VGGSound-all/ \
    --test_data_path /path/to/Flickr-SoundNet/ \
    --test_gt_path /path/to/Flickr-SoundNet/Annotations/ \
    --experiment_name vggss_144k \
    --trainset 'vggss_144k' \
    --testset 'flickr' \
    --epochs 20 \
    --batch_size 128 \
    --init_lr 0.0001