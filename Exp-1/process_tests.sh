#!/bin/bash

# 指定要遍历的文件夹
input_folder="output"
output_folder="output"

# 设置计数器
count=0

# 遍历文件夹，找到后缀为 _test.py 的文件
for file in ${input_folder}/*_test.py; do
    # # 检查计数器是否达到 N 个文件
    # if [ ${count} -ge 5 ]; then
    #     break
    # fi

    # 提取文件名前缀
    prefix=$(basename ${file} _test.py)
    echo ${prefix}
    
    # 检查输出文件是否已经存在，如果存在则跳过当前文件的处理
    if [ -e "${output_folder}/${prefix}_test.out" ]; then
        echo "Output file ${output_folder}/${prefix}_test.out already exists. Skipping file ${file}."
        continue
    fi
    
    # 在 {prefix}.py 文件前加入 from typing import *
    prefix_file="${input_folder}/${prefix}.py"
    sed -i "1i from typing import *" ${prefix_file}
    
    # 在 _test.py 文件前加入 from {prefix} import *
    echo ${file}
    sed -i "1i from ${prefix} import *" ${file}

    # 运行 _test.py 文件，输出到 output/{prefix}_test.out，错误输出到 output/{prefix}_test.err
    python ${file} > ${output_folder}/${prefix}_test.out 2> ${output_folder}/${prefix}_test.err
    
    # 打印输出信息
    echo "Test output for ${file} is saved in ${output_folder}/${prefix}_test.out"
    echo "Error output for ${file} is saved in ${output_folder}/${prefix}_test.err"
    
    echo "${count}"

     # 增加计数器
    ((count++))
done