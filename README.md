# Localizing Visual Sounds the Easy Way

Official Codebase or "Localizing Visual Sounds the Easy Way".

EZ-VSL is a simple yet effective approach for Easy Visual Sound Localization, without relying on the construction of positive and/or negative regions during training.


## Paper

[**Localizing Visual Sounds the Easy Way**]()<br>
Shentong Mo, Pedro Morgado<br>
arXiv 2022.


## Environment

To setup the environment, please simply run

```
pip install -r requirements.txt
```


## Datasets

###  Flickr-SoundNet

Data can be downloaded from [learning to localize sound source](https://github.com/ardasnck/learning_to_localize_sound_source)

###  VGG-Sound Source

Data can be downloaded from [Localizing-Visual-Sounds-the-Hard-Way](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way)

###  VGG-SS Unheard & Heard Test Data 

Data can be downloaded from [Unheard](https://github.com/stoneMo/EZ-VSL/blob/main/metadata/vggss_unheard_test.csv) and [Heard](https://github.com/stoneMo/EZ-VSL/blob/main/metadata/vggss_heard_test.csv)


## Model Zoo

We release several models pre-trained with EZ-VSL with the hope that other researchers might also benefit from them.

| Method |    Train Set   |     Test Set    |     CIoU     |  AUC  | url | args |
|:------:|:--------------:|:---------------:|:------------:|:-----:|:---:|:----:|
| EZ-VSL |   Flickr 10k   | Flickr SoundNet |     81.93    | 62.58 | [model]() | [script]() |
| EZ-VSL |   Flickr 144k  | Flickr SoundNet |     83.13    | 63.06 | [model]() | [script]() |
| EZ-VSL | VGG-Sound 144k | Flickr SoundNet |     83.94    | 63.60 | [model]() | [script]() |
| EZ-VSL |  VGG-Sound 10k |      VGG-SS     |     37.18    | 38.75 | [model]() | [script]() |
| EZ-VSL | VGG-Sound 144k |      VGG-SS     |     38.85    | 39.54 | [model]() | [script]() |
| EZ-VSL | VGG-Sound Full |      VGG-SS     |     39.34    | 39.78 | [model]() | [script]() |
| EZ-VSL |    Heard 110   |    Heard 110    |     37.25    | 38.97 | [model]() | [script]() |
| EZ-VSL |    Heard 110   |   Unheard 110   |     39.57    | 39.60 | [model]() | [script]() |

## Visualizations


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
    --init_lr 0.0001 
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


If you find this repository useful, please cite our paper:
```
@article{mo2022EZVSL,
  title={Localizing Visual Sounds the Easy Way},
  author={Mo, Shentong and Morgado, Pedro},
  journal={arXiv preprint},
  year={2022}
}
```


