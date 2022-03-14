# Localizing Visual Sounds the Easy Way

Official Codebase or "Localizing Visual Sounds the Easy Way".


## Environment

```
pip install -r requirements.txt

```


## Datasets

###  Flickr-SoundNet


Data can be downloaded from [learning to localize sound source](https://github.com/ardasnck/learning_to_localize_sound_source)


###  VGG-Sound Source

Data can be downloaded from [Localizing-Visual-Sounds-the-Hard-Way](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way)



## Train & Test


For training on our EZ-VSL, please run

```
python train.py --multiprocessing_distributed \
    --train_data_path /path/to/Flickr-all/ \
    --test_data_path /path/to/Flickr-SoundNet/ \
    --test_gt_path /path/to/Flickr-SoundNet/Annotations/ \
    --summaries_dir ./flickr \
    --trainset 'flickr' \
    --testset 'flickr' \
    --model_name 'vslnet' \
    --batch_size 128 \
    --init_lr 0.0001 \
    --epochs 20 
```


For testing and visualization, simply run

```
python test.py --test_data_path /path/to/Flickr-SoundNet/ \
    --test_gt_path /path/to/Flickr-SoundNet/Annotations/ \
    --summaries_dir /path/to/model/best.pth \
    --output_path /path/to/visualization/output \
    --testset 'flickr' \
    --iou_thres 0.5

```



