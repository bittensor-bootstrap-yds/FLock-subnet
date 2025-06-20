#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
from itertools import islice

import bittensor as bt

def parse_args():
    parser = argparse.ArgumentParser(
        description='从 JSONL 格式的文件A取前230行，与文件B的20行块组合并输出多个 JSONL 文件'
    )
    parser.add_argument('file_a',     help='文件A（JSONL）路径')
    parser.add_argument('file_b',     help='文件B（JSONL）路径')
    parser.add_argument('output_dir', help='输出目录（会被清空）')
    return parser.parse_args()

def chunked(file_obj, size):
    """把 file_obj 按 size 行打包，返回每组的列表"""
    while True:
        lines = list(islice(file_obj, size))
        if not lines:
            break
        yield lines

def main():
    args = parse_args()

    # 校验输入文件
    if not os.path.isfile(args.file_a):
        bt.logging.info(f"错误：文件A 不存在 -> {args.file_a}")
        sys.exit(1)
    if not os.path.isfile(args.file_b):
        bt.logging.info(f"错误：文件B 不存在 -> {args.file_b}")
        sys.exit(1)

    # 清空并准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.output_dir, '*')):
        try:
            os.remove(f)
        except Exception as e:
            bt.logging.info(f"清理文件失败: {f}, {e}")

    # 读取并检查文件A的前230行
    with open(args.file_a, 'r', encoding='utf-8') as fa:
        header_lines = list(islice(fa, 230))
    if len(header_lines) < 230:
        bt.logging.info(f"错误：文件A 行数不足230行（仅 {len(header_lines)} 行）")
        sys.exit(1)

    # 按20行一组读取文件B，并生成多个输出 JSONL 文件
    file_count = 0
    with open(args.file_b, 'r', encoding='utf-8') as fb:
        for idx, block in enumerate(chunked(fb, 20), start=1):
            out_path = os.path.join(
                args.output_dir,
                f'combined_{idx:03d}.jsonl'
            )
            with open(out_path, 'w', encoding='utf-8') as fo:
                # 写入文件A的230行
                fo.writelines(header_lines)
                # 写入本组文件B的20行（最后一组可能不足20行）
                fo.writelines(block)
            file_count += 1

    bt.logging.info(
        f"完成：共生成 {file_count} 个 JSONL 文件，保存在 “{args.output_dir}” 目录下。")

if __name__ == '__main__':
    bt.logging.enable_debug()
    bt.logging.info("开始处理数据集...")
    main()
