#!/bin/bash

# 指定要遍历的文件夹
base_dir="output/hf-eval-data-v4_3"
cd ${base_dir}

# 设置计数器
count=0

# 遍历文件夹，找到后缀为 .py 的文件
for file in ./*.py; do
    # 提取文件名前缀
    prefix=$(basename "$file" .py)
    echo ${prefix}
    
    # 检查输出文件是否已经存在，如果存在则跳过当前文件的处理
    if [ -e "${prefix}.out" ]; then
        echo "Output file ${prefix}.out already exists. Skipping file ${file}."
        continue
    fi
    
    # 安装必要的包
    # pip install -r ${prefix}.txt

    # 运行 _test.py 文件，输出到 output/{prefix}_test.out，错误输出到 output/{prefix}_test.err
    python ${file} > ${prefix}.out 2> ${prefix}.err
    
    # 打印输出信息
    echo "Test output for ${file} is saved in ${prefix}.out"
    echo "Error output for ${file} is saved in ${prefix}.err"
    
    echo "${count}"

     # 增加计数器
    ((count++))
done