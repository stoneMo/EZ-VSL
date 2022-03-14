


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 32 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs32_lr00001_svlnet-v2_thre001_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'contra' --model_name 'avenet' --batch_size 16 --init_lr 0.0001" --name="train_vggss_test_flickr_bs16_lr00001_avenet_10k_new"




python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre001_w1_10k"

python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.05 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr005_svlnet-v2_thre001_w1_10k"




python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./flickr --trainset 'flickr' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 32 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs32_lr00001_svlnet-v2_thre001_w1_10k_v1.4"



python sbatch_bs64.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./flickr --trainset 'flickr' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 64 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs64_lr00001_svlnet-v2_thre001_w1_10k"







python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-test/ --summaries_dir ./vggss_tvgg --trainset 'vggss' --testset 'vggss' --loss_type 'contra' --model_name 'avenet' --batch_size 16 --init_lr 0.0001" --name="train_vggss_test_vggss_bs16_lr00001_avenet_10k"



python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-test/ --summaries_dir ./vggss_tvgg --trainset 'vggss' --testset 'vggss' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 32 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_vggss_bs32_lr00001_svlnet-v2_thre001_w1_10k"




# thresh 


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 1 --std_weight 0" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre1_w0_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 2 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre2_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.01 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre001_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.1 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre01_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.05 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre0005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.001 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre0001_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0001 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00001_w1_10k"



# bs

python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 32 --init_lr 0.0001 --std_thres 0.0005 --std_weight 10" --name="train_vggss_test_flickr_bs32_lr00001_svlnet-v2_thre00005_w10_10k"


python sbatch_bs64.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 64 --init_lr 0.0001 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs64_lr00001_svlnet-v2_thre00005_w1_10k"





# std weight


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 100" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w100_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 10" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w10_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 0.1" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w01_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 0.01" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w001_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0001 --std_thres 0.0005 --std_weight 0.001" --name="train_vggss_test_flickr_bs16_lr00001_svlnet-v2_thre00005_w0001_10k"



# learning rate

python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.0005 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr00005_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.001 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr0001_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.005 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr0005_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.01 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr001_svlnet-v2_thre00005_w1_10k"


python sbatch.py launch --cmd="python train.py --multiprocessing_distributed --train_data_path /project_data/held/jianrenw/multimodal_feature/datasets/VGGSound-10k/ --test_data_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/ --test_gt_path /project_data/held/jianrenw/multimodal_feature/datasets/Flickr-SoundNet/Annotations/ --summaries_dir ./vggss --trainset 'vggss' --testset 'flickr' --loss_type 'ce+std' --model_name 'svlnet-v2' --batch_size 16 --init_lr 0.05 --std_thres 0.0005 --std_weight 1" --name="train_vggss_test_flickr_bs16_lr005_svlnet-v2_thre00005_w1_10k"

