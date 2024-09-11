# 1.下载项目

git clone https://github.com/hpcaitech/Open-Sora

cd Open-Sora

# 2.构建镜像
docker build -t opensora .

# 3.运行容器
docker run  -dt --name opensora --restart=always --shm-size 100G  --gpus all -v /root/sora/Open-Sora:/workspace/Open-Sora opensora

docker stop opensora

docker rm opensora

docker exec -it opensora bash

# 4.容器内安装软件
pip install scenedetect

pip install imageio_ffmpeg

pip install git+https://github.com/openai/CLIP.git

pip install git+https://github.com/haotian-liu/LLaVA.git

下载llava-v1.6-mistral-7b 

mkdir /workspace/Open-Sora/model_ckpt

cd /workspace/Open-Sora/model_ckpt

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs

git lfs install

git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b

git lfs clone https://huggingface.co/DeepFloyd/t5-v1_1-xxl

# 5.数据处理

## 5.1可以从youtube上下载视频 也下载数据集
mkdir /workspace/Open-Sora/data/

mkdir /workspace/Open-Sora/data/data_vedio

pip install git+https://github.com/yt-dlp/yt-dlp.git

yt-dlp https://www.youtube.com/watch?v=srRaCkQVvR0

下载视频 -> /workspace/Open-Sora/data/data_vedio
需要把下载视频改成MP4格式

## 5.2处理视频
申明环境变量
export ROOT_VIDEO="/workspace/Open-Sora/data/data_vedio"

export ROOT_CLIPS="/workspace/Open-Sora/data/clips"

export ROOT_META="/workspace/Open-Sora/data"

5.2.1.1 从视频文件夹创建元文件
python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

5.2.1.2 获取视频信息并删除损坏的视频
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1

5.2.2.1 检测场景
python -m tools.scene_cut.scene_detect ${ROOT_META}/meta_info_fmin1.csv

5.2.2.2 根据场景将视频剪辑成片段
python -m tools.scene_cut.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}

5.2.2.3 为视频剪辑创建元文件
python -m tools.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv

5.2.2.4 获取剪辑信息并删除损坏的剪辑。
python -m tools.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1

5.2.3.1 预测美学分数
torchrun --nproc_per_node 8 -m tools.scoring.aesthetic.inference \
  ${ROOT_META}/meta_clips_info_fmin1.csv \
  --bs 1024 \
  --num_workers 16

No such file or directory: 'pretrained_models/aesthetic.pth' 解决方案

mkdir /workspace/Open-Sora/model_ckpt/pretrained_models/

cd /workspace/Open-Sora/model_ckpt/pretrained_models/

wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O /workspace/Open-Sora/model_ckpt/pretrained_models/aesthetic.pth

修改inference.py代码 

151行改为model.mlp.load_state_dict(torch.load("/workspace/Open-Sora/model_ckpt/pretrained_models/aesthetic.pth", map_location=device))

cd /workspace/Open-Sora/

pip install -e .


5.2.3.2 过滤美学分数低于5的视频
python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes.csv --aesmin 5



5.2.4.1 生成视频说明 

torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava \
  ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv \
  --dp-size 8 \
  --tp-size 1 \
  --model-path /workspace/Open-Sora/model_ckpt/llava-v1.6-mistral-7b \
  --prompt video

5.2.4.2 合并视频说明结果
python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0_caption_part*.csv --output ${ROOT_META}/meta_clips_caption.csv


5.2.4.3 清理视频说明
python -m tools.datasets.datautil \
  ${ROOT_META}/meta_clips_caption.csv \
  --clean-caption \
  --refine-llm-caption \
  --remove-empty-caption \
  --output ${ROOT_META}/meta_clips_caption_cleaned.csv

5.2.5.1 合并视频说明和美学分数

python data_merge.py

data_merge.py内容为：
import pandas as pd

df  = pd.read_csv('meta_clips_caption_cleaned.csv')

df1 = pd.read_csv('meta_clips_info_fmin1_aes.csv')

df = df[['path','text']].merge(df1)

df.to_csv('meta_clips_caption_text.csv')


# 6.重新运行容器
docker stop opensora

docker rm opensora

docker run  -dt --name opensora --restart=always --shm-size 100G  --gpus all -v /root/sora/Open-Sora:/workspace/Open-Sora opensora

docker exec -it opensora bash

export ROOT_VIDEO="/workspace/Open-Sora/data/data_vedio"
export ROOT_CLIPS="/workspace/Open-Sora/data/clips"
export ROOT_META="/workspace/Open-Sora/data"

# 7.训练
torchrun --standalone --nproc_per_node 8 scripts/train.py configs/opensora-v1-2/train/stage3.py \
 --data-path ${ROOT_META}/meta_clips_caption_text.csv

## one node
torchrun --standalone --nproc_per_node 8 scripts/train.py \
    configs/opensora-v1-2/train/stage1.py --data-path ${ROOT_META}/meta_clips_caption_text.csv --ckpt-path YOUR_PRETRAINED_CKPT
## multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/stage1.py ---data-path ${ROOT_META}/meta_clips_caption_text.csv --ckpt-path YOUR_PRETRAINED_CKPT

# 8.推理
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall"
