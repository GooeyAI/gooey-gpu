#!/usr/bin/env bash

set -ex

DIR=$PWD

CHECKPOINTS_DIR=$HOME/.cache/gooey-gpu/checkpoints
cd $CHECKPOINTS_DIR

wget -c -O wav2lip_gan.pth 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA'
wget -c -O mobilenet.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth'
wget -c -O resnet50.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth'

pip install gdown

python3 $DIR/retro/U-2-Net/setup_model_weights.py
python3 -c 'import gdown; gdown.download("https://drive.google.com/uc?id=1XHIzgTzY5BQHw140EDIgwIb53K659ENH", "isnet-general-use.pth")'

mkdir sadtalker
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -O  sadtalker/mapping_00109-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -O  sadtalker/mapping_00229-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O  sadtalker/SadTalker_V0.0.2_256.safetensors
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -O  sadtalker/SadTalker_V0.0.2_512.safetensors

mkdir -p sadtalker/gfpgan/weights
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -O sadtalker/gfpgan/weights/alignment_WFLW_4HG.pth
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O sadtalker/gfpgan/weights/detection_Resnet50_Final.pth
wget -nc https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O sadtalker/gfpgan/weights/GFPGANv1.4.pth
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O sadtalker/gfpgan/weights/parsing_parsenet.pth
