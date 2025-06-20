#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List

import bittensor as bt

def parse_args():
    parser = argparse.ArgumentParser(
        description='遍历指定文件夹下所有 .jsonl 文件，并校验其内容格式'
    )
    parser.add_argument(
        'input_dir',
        help='包含 .jsonl 文件的目录路径'
    )
    return parser.parse_args()

def find_jsonl_files(directory: str) -> List[str]:
    """返回目录下所有以 .jsonl 结尾的文件完整路径列表（不递归子目录）。"""
    return [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith('.jsonl') and os.path.isfile(os.path.join(directory, fname))
    ]

def check(file: str) -> None:
    """
    校验单个 .jsonl 文件的存在性、非空性，以及每行 JSON 结构：
      - 顶层必须包含 'system'
      - 'conversations' 必须是非空列表
      - conversations 中交替出现 user/assistant
    """
    if not os.path.exists(file):
        raise ValueError(f"File {file} does not exist.")
    if not file.endswith('.jsonl'):
        raise ValueError(f"File {file} is not a .jsonl file.")
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"File {file} is empty.")

    for idx, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {idx} in {file} is not valid JSON: {e}")

        # 校验字段
        if 'system' not in data or data['system'] is None:
            raise ValueError(f"Line {idx} in {file}: missing or null 'system'")
        if 'conversations' not in data or not isinstance(data['conversations'], list) or not data['conversations']:
            raise ValueError(f"Line {idx} in {file}: 'conversations' must be a non-empty list")

        last_role = None
        for msg in data['conversations']:
            if not isinstance(msg, dict):
                raise ValueError(f"Line {idx} in {file}: each conversation entry must be a dict")
            role = msg.get('role')
            content = msg.get('content')
            if role not in ('user', 'assistant'):
                raise ValueError(f"Line {idx} in {file}: invalid role '{role}'")
            if role == last_role:
                raise ValueError(f"Line {idx} in {file}: consecutive messages have the same role '{role}'")
            if not isinstance(content, str):
                raise ValueError(f"Line {idx} in {file}: content must be a string")
            last_role = role

    bt.logging.info(f"[OK] {file}")

def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        bt.logging.info(f"错误：{args.input_dir} 不是一个有效目录")
        sys.exit(1)

    jsonl_files = find_jsonl_files(args.input_dir)
    if not jsonl_files:
        bt.logging.info(f"目录 {args.input_dir} 下未找到任何 .jsonl 文件")
        sys.exit(1)

    for filepath in jsonl_files:
        try:
            check(filepath)
        except Exception as e:
            bt.logging.info(f"[ERROR] {e}")
            # 如果希望遇到第一个错误就退出，取消下面两行的注释：
            # sys.exit(1)
    bt.logging.info("所有文件校验完成。")

if __name__ == "__main__":
    bt.logging.enable_debug()
    bt.logging.info("开始校验 .jsonl 文件...")
    if len(sys.argv) < 2:
        bt.logging.info("请提供包含 .jsonl 文件的目录路径。")
        sys.exit(1)
    main()
