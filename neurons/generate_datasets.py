#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
import random

import bittensor as bt

def parse_args():
    parser = argparse.ArgumentParser(
        description='从 JSONL 格式的文件A随机取230行，与文件B随机取20行组合并生成32个 JSONL 文件'
    )
    parser.add_argument('file_a',     help='文件A（JSONL）路径')
    parser.add_argument('file_b',     help='文件B（JSONL）路径')
    parser.add_argument('output_dir', help='输出目录（会被清空）')
    return parser.parse_args()

def main():
    args = parse_args()

    # 校验输入文件
    if not os.path.isfile(args.file_a):
        bt.logging.error(f"文件A 不存在: {args.file_a}")
        sys.exit(1)
    if not os.path.isfile(args.file_b):
        bt.logging.error(f"文件B 不存在: {args.file_b}")
        sys.exit(1)

    # 读取所有行
    with open(args.file_a, 'r', encoding='utf-8') as fa:
        lines_a = fa.readlines()
    with open(args.file_b, 'r', encoding='utf-8') as fb:
        lines_b = fb.readlines()

    # 检查行数是否满足随机抽样
    if len(lines_a) < 230:
        bt.logging.error(f"文件A 行数不足 230 行（仅 {len(lines_a)} 行）")
        sys.exit(1)
    if len(lines_b) < 20:
        bt.logging.error(f"文件B 行数不足 20 行（仅 {len(lines_b)} 行）")
        sys.exit(1)

    # 清空并准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.output_dir, '*')):
        try:
            os.remove(f)
        except Exception as e:
            bt.logging.exception(f"清理文件失败: {f} -> {e}")

    # 生成 64 个文件
    for idx in range(1, 65):
        # 随机抽样
        sample_a = random.sample(lines_a, 230)
        sample_b = random.sample(lines_b, 20)

        out_path = os.path.join(
            args.output_dir,
            f'combined_{idx:03d}.jsonl'
        )
        with open(out_path, 'w', encoding='utf-8') as fo:
            fo.writelines(sample_a)
            fo.writelines(sample_b)

        bt.logging.info(f"已生成 {out_path}")

    bt.logging.info(f"完成：共生成 64 个 JSONL 文件，保存在 “{args.output_dir}” 目录下。")

if __name__ == '__main__':
    bt.logging.enable_debug()
    bt.logging.info("开始处理数据集...")
    main()
