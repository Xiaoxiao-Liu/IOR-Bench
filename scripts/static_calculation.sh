#!/bin/bash

# 如果当前环境不是 iorbench，则激活
if [ "$CONDA_DEFAULT_ENV" != "iorbench" ]; then
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate iorbench
fi

# 切换到项目根目录
cd /absolute/path/to/IOR-Bench


# 使用 -m 方式运行模块
python -m src.post_processing.static_calc --model_name "gpt-4o" 