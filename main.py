"""单图/目录批量识别。python main.py --help"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from pipeline import Reader, read_image, runtime_arguments, save_result
from rules import DEFAULT_BATCH_PATTERN

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_roi(value):
    try:
        values = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI 格式为 x,y,width,height") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI 必须包含四个整数")
    return values


def main():
    parser = argparse.ArgumentParser(description="包装喷码识别（可配置运行设备 / PaddleOCR 3.x）")
    parser.add_argument("input", type=Path, help="单张图片或图片目录，目录模式不递归")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--mode", choices=["raw", "enhance", "auto"], default="auto")
    parser.add_argument("--roi", type=parse_roi, help="原图经 EXIF 方向修正后的 x,y,width,height")
    parser.add_argument("--angle", type=float, default=0, help="逆时针旋转角度，可用于 90/180 度倒置图")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch-pattern", default=DEFAULT_BATCH_PATTERN)
    runtime_arguments(parser)
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in EXTENSIONS) if args.input.is_dir() else [args.input]
    if not paths or not all(p.is_file() for p in paths):
        parser.error("未找到输入图片")
    run = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    reader = Reader(args.threads, args.device, args.mkldnn, args.max_side)
    failures, summary = 0, []
    for index, path in enumerate(paths):
        try:
            image = read_image(path)
            result = reader.recognize(image, mode=args.mode, roi=args.roi, angle=args.angle, threshold=args.threshold, batch_pattern=args.batch_pattern)
            result["input"] = str(path.resolve())
            save_result(run / f"{index:04d}_{path.stem}", image, result)
            row = {"file": str(path), "decision": result["decision"], "production_date": result["fields"]["production_date"]["value"], "batch": result["fields"]["batch"]["value"], "elapsed_ms": result["elapsed_ms"]}
        except Exception as exc:
            failures += 1
            row = {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (run / "summary.json").write_text(json.dumps({"model_init_ms": reader.init_ms, "runtime": reader.runtime_info, "files": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{run.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
