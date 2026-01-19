#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from face_mosaic import FaceMosaicProcessor


def process_single_video(processor, input_path, output_dir, overwrite):
    """
    处理单个视频文件
    """
    try:
        input_path_obj = Path(input_path)

        # 确定输出路径
        if output_dir:
            # 计算相对于根目录的路径（这里假设脚本知道根目录，或者我们传入root）
            # 由于在多线程中传递 root 比较麻烦，我们可以约定 input_path 是绝对路径
            # 这里的逻辑稍微复杂，需要在外部计算好 dest_path 或者传入 root
            # 为了简化，我们在这个函数里不计算相对路径，而是由调用者传入 dest_path
            pass

        # 这里为了灵活，我们重新定义函数签名
        # 实际逻辑在 process_batch 中处理路径计算
        return False, "Function signature mismatch"

    except Exception as e:
        return False, str(e)


def process_task(args):
    """
    包装函数用于多线程执行
    args: (input_path, output_path, overwrite, processor_config)
    """
    input_path, final_output_path, overwrite, processor_config = args

    # 每个线程创建一个新的处理器实例以确保线程安全
    processor = FaceMosaicProcessor(**processor_config)

    try:
        # 1. 预先检测人脸
        # 使用默认的采样间隔（30帧）进行快速检测
        has_face, _ = processor.check_video_has_face(input_path)

        if not has_face:
            # 如果没有检测到人脸
            # 检查是否在同一目录下（包括覆盖模式和默认的_processed模式）
            is_same_dir = os.path.dirname(os.path.abspath(input_path)) == os.path.dirname(
                os.path.abspath(final_output_path))

            if is_same_dir:
                # 如果是在同一目录下（无论是覆盖还是生成新文件），直接跳过，不创建/修改文件
                return True, f"跳过 (无人脸): {Path(input_path).name}"
            else:
                # 输出到新目录模式：直接复制原文件，保持目录结构完整
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                if not os.path.exists(final_output_path):
                    shutil.copy2(input_path, final_output_path)
                return True, f"复制 (无人脸): {Path(input_path).name}"

        # 2. 检测到人脸，开始打马赛克处理
        temp_output_path = final_output_path

        # 如果是覆盖模式或者输出路径等于输入路径
        if overwrite or os.path.abspath(input_path) == os.path.abspath(final_output_path):
            # 创建临时文件
            temp_output_path = str(Path(final_output_path).with_name(
                f".tmp_{Path(final_output_path).name}"))

        # 确保输出目录存在
        os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)

        # 处理视频
        success = processor.process_video(input_path, temp_output_path)

        if success:
            if temp_output_path != final_output_path:
                # 如果是临时文件，移动到最终位置（覆盖）
                if os.path.exists(final_output_path):
                    os.remove(final_output_path)
                os.rename(temp_output_path, final_output_path)
            return True, input_path
        else:
            # 清理失败的临时文件
            if temp_output_path != final_output_path and os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            return False, input_path

    except Exception as e:
        return False, f"{input_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="GM100 数据集人脸马赛克批量处理工具")
    parser.add_argument("--root", default="/nas/data/GM100",
                        help="数据根目录 (默认: /nas/data/GM100)")
    parser.add_argument(
        "--output", help="输出根目录。如果指定，将镜像目录结构。如果不指定，默认在原文件旁生成 _processed 文件。")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖原始文件。注意：此操作不可逆！")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数 (默认: 4)")
    parser.add_argument(
        "--pattern", default="task_*/cobot/page_data/trainset/episode_*/cam-*.mp4", help="文件匹配模式")

    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)

    # 构造搜索模式
    search_pattern = os.path.join(root_dir, args.pattern)
    print(f"🔍 正在搜索文件: {search_pattern}")

    # 使用 glob 查找文件
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"❌ 未找到匹配的文件。请检查路径: {root_dir}")
        return

    print(f"📁 找到 {len(files)} 个视频文件")

    # 准备任务参数
    tasks = []
    processor_config = {
        'confidence': 0.7,
        'mosaic_size': 20,
        'preserve_audio': False
    }

    for input_path in files:
        input_path_obj = Path(input_path)

        if args.output:
            # 镜像目录结构
            try:
                rel_path = input_path_obj.relative_to(root_dir)
                final_output_path = str(Path(args.output) / rel_path)
            except ValueError:
                # 如果文件不在 root_dir 下（不太可能，因为是用 glob 在 root_dir 下找的），回退到默认
                final_output_path = str(input_path_obj.with_stem(
                    f"{input_path_obj.stem}_processed"))
        elif args.overwrite:
            final_output_path = str(input_path_obj)
        else:
            # 默认：同目录下加后缀
            final_output_path = str(input_path_obj.with_stem(
                f"{input_path_obj.stem}_processed"))

        tasks.append((input_path, final_output_path,
                     args.overwrite, processor_config))

    print(f"🚀 开始处理，使用 {args.workers} 个线程...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交任务
        future_to_file = {executor.submit(
            process_task, task): task[0] for task in tasks}

        # 进度条
        with tqdm(total=len(tasks), desc="总进度") as pbar:
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        tqdm.write(f"❌ 失败: {msg}")
                except Exception as e:
                    fail_count += 1
                    tqdm.write(f"❌ 异常: {input_file} - {e}")

                pbar.update(1)
                pbar.set_postfix({"成功": success_count, "失败": fail_count})

    print("\n" + "="*60)
    print(f"📊 处理完成")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print("="*60)


if __name__ == "__main__":
    main()
