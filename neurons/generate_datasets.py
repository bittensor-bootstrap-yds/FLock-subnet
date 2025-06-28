#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 文件夹A 中读取所有 .jsonl 文件，去重（并集），
随机挑选 250 条生成样本文件。
每个样本文件：
  - 与 任意 原始文件 的重复条数 <= 199
  - 与 上一个样本文件 的重复条数 <= 199
生成样本个数最多 12 个，至少 1 个，写入 文件夹B。
"""
import argparse
import os
import sys
import glob
import random
import bittensor as bt


def parse_args():
    parser = argparse.ArgumentParser(
        description='基于 JSONL 并集，随机抽 250 条生成样本文件'
    )
    parser.add_argument('dir_a', help='输入目录A，包含多个 .jsonl 文件')
    parser.add_argument('dir_b', help='输出目录B，样本文件写入此目录')
    parser.add_argument('--max-files', type=int, default=12,
                        help='最多生成的样本文件个数（默认 12）')
    return parser.parse_args()


def load_and_union(dir_a):
    """
    读取 dir_a 中所有 .jsonl 文件，返回：
      - union_lines: 去重后的所有行列表
      - orig_sets: 每个原始文件的行集合列表
    """
    paths = glob.glob(os.path.join(dir_a, '*.jsonl'))
    if not paths:
        bt.logging.error(f"目录 A 中没有找到 .jsonl 文件: {dir_a}")
        sys.exit(1)
    orig_sets = []
    union_set = set()
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f if l.strip()]
        s = set(lines)
        orig_sets.append(s)
        union_set.update(s)
        bt.logging.debug(f"Loaded {len(lines)} lines, unique {len(s)} from {p}")
    union_lines = list(union_set)
    bt.logging.info(f"总并集行数: {len(union_lines)}")
    return union_lines, orig_sets


def sample_valid(union_lines, orig_sets, prev_set=None):
    """
    从 union_lines 随机抽 250 条，验证：
      - 与任意 orig_set 的交集 <=199
      - 与 prev_set 的交集 <=199（若 prev_set 不为 None）
    返回：新样本集 set(lines)
    """
    for trial in range(1000):
        sample = set(random.sample(union_lines, 250))
        # 验证与各原始集合的重复 <=199
        ok = True
        for s in orig_sets:
            if len(sample & s) > 199:
                ok = False
                break
        if not ok:
            continue
        # 验证与上一次样本的重复 <=199
        if prev_set is not None and len(sample & prev_set) > 199:
            continue
        return sample
    bt.logging.error("多次尝试仍无法生成满足条件的样本集，请检查数据或放宽条件。")
    sys.exit(1)


def main():
    args = parse_args()

    # 准备输出目录
    os.makedirs(args.dir_b, exist_ok=True)
    for f in glob.glob(os.path.join(args.dir_b, '*')):
        try:
            os.remove(f)
        except Exception:
            pass

    # 读取并集及原始集合
    union_lines, orig_sets = load_and_union(args.dir_a)

    prev_set = None
    count = 0
    for idx in range(1, args.max_files + 1):
        bt.logging.info(f"正在生成第 {idx} 个样本...")
        sample = sample_valid(union_lines, orig_sets, prev_set)
        out_path = os.path.join(args.dir_b, f'sample_{idx:02d}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as fo:
            for line in sample:
                fo.write(line + '\n')
        bt.logging.info(f"已写入样本文件: {out_path}")
        prev_set = sample
        count += 1

    bt.logging.info(f"完成：共生成 {count} 个样本文件，保存在 '{args.dir_b}'。")


if __name__ == '__main__':
    bt.logging.enable_debug()
    bt.logging.info("开始生成样本文件...")
    main()
