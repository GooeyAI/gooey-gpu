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
python3 -c 'import gdown; gdown.download("https://drive.google.com/uc?id=1nV57qKuy--d5u1yvkng9aXW1KS4sOpOi", "isnet-general-use.pth")'
