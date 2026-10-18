"""生成可复现的合成烟雾测试集；不能用其准确率声称真实工业效果。"""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import rotate, save_image


def generate(output, font_path):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    font = ImageFont.truetype(str(font_path), 38)
    labels = []

    def add(name, bgr, text, day, batch, group, split, condition, decision="PASS"):
        save_image(output / "images" / f"{name}.png", bgr)
        labels.append(dict(image=f"images/{name}.png", text=text, production_date=day, batch=batch,
                           group=group, split=split, condition=condition, decision=decision, source="synthetic_pillow_seed42"))

    def render(lines):
        image = Image.new("RGB", (900, 300), (246, 243, 233))
        draw = ImageDraw.Draw(image)
        for i, text in enumerate(lines):
            draw.text((65, 55 + i * 72), text, fill=(28, 32, 30), font=font)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    for i in range(8):
        day = date(2026, 1, 10) + timedelta(days=i * 27)
        batch = f"A{day:%y%m%d}B{i + 1:02d}"
        lines = [f"MFG: {day:%Y.%m.%d}", f"LOT: {batch}"]
        if i % 2:
            lines[0] = f"生产日期: {day:%Y年%m月%d日}"
            lines[1] = f"批号: {batch}"
        base = render(lines)
        variants = {"clean": base, "tilt": rotate(base, (-1 if i % 2 else 1) * 12)[0],
                    "blur": cv2.GaussianBlur(base, (7, 7), 1.6),
                    "low_contrast": np.clip(base.astype(float) * 0.24 + 166, 0, 255).astype(np.uint8),
                    "noise": np.clip(base.astype(float) + rng.normal(0, 15, base.shape), 0, 255).astype(np.uint8)}
        dots = base.copy()
        # 规则网格擦除模拟断墨，仅是数字扰动，不是物理喷墨成像模型。
        dots[::4, :] = (246, 243, 233)
        dots[:, ::4] = (246, 243, 233)
        variants["broken_ink"] = dots
        for condition, image in variants.items():
            add(f"s{i:02d}_{condition}", image, "\n".join(lines), day.isoformat(), batch, f"s{i:02d}", "dev" if i < 4 else "test", condition)
    negatives = [
        ("invalid_date", ["MFG: 2026.02.30", "LOT: A260230B01"], None, "A260230B01", "dev"),
        ("expiry_only", ["EXP: 2027.10.18", "LOT: A261018B01"], None, "A261018B01", "dev"),
        ("missing_batch", ["MFG: 2026.10.18"], "2026-10-18", None, "test"),
        ("invalid_calendar", ["MFG: 2025.02.29", "LOT: A250229B01"], None, "A250229B01", "test"),
    ]
    for name, lines, day, batch, split in negatives:
        add(name, render(lines), "\n".join(lines), day, batch, name, split, name, "REVIEW")
    (output / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in labels) + "\n", encoding="utf-8")
    print(f"已生成 {len(labels)} 张合成图片：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"), help="支持中文的字体文件")
    args = parser.parse_args()
    generate(args.output, args.font)
