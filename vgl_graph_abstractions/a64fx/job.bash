#!/bin/bash

if [ -e ../vars_global.bash ]; then
    echo ../vars_global.bash exists
    source ../vars_global.bash
fi
if [ -e ./vars.bash ]; then
    echo ./vars.bash exists
    source ./vars.bash
fi

export PATH=/home/z44377r/arm_gcc/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/z44377r/arm_gcc/lib/:/home/z44377r/arm_gcc/lib64/

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
./saxpy.bin