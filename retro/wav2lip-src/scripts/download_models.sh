#!/usr/bin/env bash

set -ex

CHECKPOINTS_DIR=$HOME/.cache/gooey-gpu/checkpoints

wget -c -O $CHECKPOINTS_DIR/wav2lip_gan.pth 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA'
wget -c -O $CHECKPOINTS_DIR/mobilenet.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth'
wget -c -O $CHECKPOINTS_DIR/resnet50.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth'
