#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 dir_a 中的原始 .jsonl 文件生成样本文件。
生成样本文件数量与原始文件数量一致。
每个样本文件 sample_{i}.jsonl:
  - 随机选取一个原始文件的 160 条
  - 从其余原始文件中选取 60 条
  - 从 eval_data_file 中选取 30 条
  共计 250 条，保证无重复。

生成完成后，使用 count_similar 方法检查：
  1) 各样本文件之间的相似度
  2) 每个样本文件与 dir_a 中每个原始文件的相似度
  3) 原始文件之间的相似度
并打印每个样本文件的行数。
"""
import argparse
import os
import sys
import glob
import random
import json
import bittensor as bt

def parse_args():
    parser = argparse.ArgumentParser(
        description='基于 dir_a 文件生成样本文件'
    )
    parser.add_argument('dir_a', help='输入目录A，包含多个 .jsonl 文件')
    parser.add_argument('eval_data_file', help='评估数据文件，包含至少 30 条 .jsonl 格式行')
    parser.add_argument('dir_b', help='输出目录B，样本文件写入此目录')
    return parser.parse_args()


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def load_eval(eval_file):
    if not os.path.isfile(eval_file):
        bt.logging.error(f"评估数据文件不存在: {eval_file}")
        sys.exit(1)
    lines = load_jsonl(eval_file)
    unique = set(lines)
    bt.logging.info(f"评估数据总行数: {len(lines)}, 唯一: {len(unique)}")
    if len(unique) < 30:
        bt.logging.error("评估数据文件需至少包含 30 条唯一行")
        sys.exit(1)
    return list(unique)


def count_similar(jsonl1, jsonl2):
    set1 = set(json.dumps(json.loads(item), sort_keys=True) for item in jsonl1)
    set2 = set(json.dumps(json.loads(item), sort_keys=True) for item in jsonl2)
    return len(set1 & set2)


def main():
    args = parse_args()
    os.makedirs(args.dir_b, exist_ok=True)
    # 清空输出目录
    for f in glob.glob(os.path.join(args.dir_b, '*.jsonl')):
        try: os.remove(f)
        except: pass

    # 加载原始文件内容
    orig_paths = glob.glob(os.path.join(args.dir_a, '*.jsonl'))
    if not orig_paths:
        bt.logging.error(f"目录 A 中没有找到 .jsonl 文件: {args.dir_a}")
        sys.exit(1)
    orig_jsonls = [(os.path.basename(p), load_jsonl(p)) for p in orig_paths]

    # 检查原始文件数量
    num_files = len(orig_jsonls)
    bt.logging.info(f"原始文件数量: {num_files}")

    # 加载评估数据
    eval_lines = load_eval(args.eval_data_file)

    # 随机打乱原始文件顺序，用于对应样本文件
    random.shuffle(orig_jsonls)
    generated = []

    for idx, (orig_name, orig_data) in enumerate(orig_jsonls, start=1):
        bt.logging.info(f"生成样本 {idx}/{num_files}, 基于原始文件: {orig_name}")
        base_set = set(orig_data)
        if len(base_set) < 160:
            bt.logging.error(f"原始文件 {orig_name} 行数不足 160，无法抽样")
            sys.exit(1)
        # 抽取 160 条基础数据
        base_sample = set(random.sample(orig_data, 160))

        # 其余文件的联合行集合
        other_union = set()
        for name, data in orig_jsonls:
            if name != orig_name:
                other_union.update(data)
        other_candidates = other_union - base_sample
        if len(other_candidates) < 60:
            bt.logging.error(f"其余文件可抽行数不足 60，无法抽样 for {orig_name}")
            sys.exit(1)
        other_sample = set(random.sample(list(other_candidates), 60))

        # 抽取 30 条评估数据
        eval_candidates = set(eval_lines) - base_sample - other_sample
        if len(eval_candidates) < 30:
            bt.logging.error("评估数据可抽行数不足 30，无法抽样")
            sys.exit(1)
        eval_sample = set(random.sample(list(eval_candidates), 30))

        # 合并并检查总数
        total = base_sample | other_sample | eval_sample
        if len(total) != 250:
            bt.logging.error(f"样本总行数不等于250，仅 {len(total)} 行: {orig_name}")
            sys.exit(1)

        # 写入样本文件
        out_path = os.path.join(args.dir_b, f'sample_{idx}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as fo:
            for line in total:
                fo.write(line + '\n')
        bt.logging.info(f"已写入样本文件: {out_path}")
        generated.append(out_path)

    bt.logging.info(f"完成：共生成 {len(generated)} 个样本文件，保存在 '{args.dir_b}'。" )
    bt.logging.info("开始相似度和行数检查...")

    # 加载生成样本内容
    sample_jsonls = [(os.path.basename(p), load_jsonl(p)) for p in generated]

    # 打印样本文件行数
    for sn, sd in sample_jsonls:
        bt.logging.info(f"样本文件行数: {sn} = {len(sd)}")

    # 样本文件之间相似度
    for i in range(len(sample_jsonls)):
        ni, di = sample_jsonls[i]
        for j in range(i+1, len(sample_jsonls)):
            nj, dj = sample_jsonls[j]
            sim = count_similar(di, dj)
            bt.logging.info(f"样本相似度: {ni} ↔ {nj} = {sim}")

    # 样本文件与原始文件相似度
    for sn, sd in sample_jsonls:
        for on, od in orig_jsonls:
            sim = count_similar(sd, od)
            bt.logging.info(f"样本 vs 原始: {sn} vs {on} = {sim}")

    # 原始文件之间相似度
    for i in range(len(orig_jsonls)):
        ni, di = orig_jsonls[i]
        for j in range(i+1, len(orig_jsonls)):
            nj, dj = orig_jsonls[j]
            sim = count_similar(di, dj)
            bt.logging.info(f"原始相似度: {ni} ↔ {nj} = {sim}")

if __name__ == '__main__':
    bt.logging.enable_debug()
    bt.logging.info("开始生成并检查样本文件...")
    main()
