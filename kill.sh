#!/bin/bash

# 查找所有使用 NVIDIA 设备的 Python 进程
PIDS=$(lsof /dev/nvidia* | grep python | awk '{print $2}' | sort -u)

# 如果找到进程，则杀死它们
if [ -n "$PIDS" ]; then
    echo "Killing the following Python processes using NVIDIA devices:"
    echo "$PIDS"
    kill -9 $PIDS
else
    echo "No Python processes using NVIDIA devices found."
fi