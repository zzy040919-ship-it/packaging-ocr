"""用实际评估输出生成演示视频；是结果讲解片，不是桌面操作录屏。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline import ROOT, font


def make_demo(report_path, output):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    records = report["modes"]["auto"]["records"]
    names = ["s04_clean.png", "s04_tilt.png", "s05_blur.png", "s05_broken_ink.png", "s04_broken_ink.png", "invalid_calendar.png"]
    frames = []
    for page, name in enumerate(names, 1):
        index = next(i for i, r in enumerate(records) if Path(r["image"]).name == name)
        record = records[index]
        result = json.loads((report_path.parent / "auto" / f"{index:04d}" / "result.json").read_text(encoding="utf-8"))
        frame = Image.new("RGB", (1280, 720), "#f4f3ef")
        draw = ImageDraw.Draw(frame)
        draw.text((40, 24), "包装喷码识别 / PaddleOCR", font=font(32), fill="#202923")
        draw.text((40, 76), f"{page}/6  {record['condition']} · {name}", font=font(20), fill="#526158")
        draw.text((40, 111), "合成样例 · 真实模型推理结果 · 非工业准确率验证", font=font(18), fill="#875516")
        preview = Image.open(report_path.parent / "auto" / f"{index:04d}" / "annotated.png").convert("RGB")
        preview.thumbnail((1200, 345))
        frame.paste(preview, ((1280 - preview.width) // 2, 158))
        color = "#167454" if result["decision"] == "PASS" else "#a45812"
        text = "通过规则检查" if result["decision"] == "PASS" else "需要人工复核"
        draw.text((40, 515), f"{text}  |  {result['variant']}  |  {result['elapsed_ms']:.0f} ms", font=font(26), fill=color)
        for j, (key, title) in enumerate([("production_date", "生产日期"), ("batch", "批次号")]):
            field = result["fields"][key]
            draw.text((40, 562 + j * 38), f"{title}：{field['value'] or '未确认'}  [{field['status']}]", font=font(23), fill="#202923")
        draw.text((40, 665), "预处理 + OCR + 字段校验；计时不含模型加载。PASS 不等于产品质量合格。", font=font(17), fill="#526158")
        frames.append(frame)
    writer = cv2.VideoWriter(str(output / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("MP4 编码器不可用")
    try:
        for frame in frames:
            bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            for _ in range(40):
                writer.write(bgr)
    finally:
        writer.release()
    frames[1].save(output / "preview.png")
    frames[0].save(output / "demo.gif", save_all=True, append_images=frames[1:], duration=4000, loop=0)
    print(f"演示文件：{output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation" / "final" / "metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "demo")
    args = parser.parse_args()
    make_demo(args.report, args.output)
