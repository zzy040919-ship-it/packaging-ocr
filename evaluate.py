"""同一测试集比较 raw 与 auto；所有失败计入分母，保留逐图输出。"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline import Reader, read_image, save_result
from rules import normalize


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalized_text(text):
    return "".join(normalize(text).split())


def metrics(records):
    result = {"count": len(records), "errors": sum("error" in r for r in records)}
    for field in ("production_date", "batch"):
        eligible = [r for r in records if r["expected"][field] is not None]
        result[field + "_exact_match"] = sum(r.get(field) == r["expected"][field] for r in eligible) / len(eligible) if eligible else None
        result[field + "_denominator"] = len(eligible)
    result["both_fields_exact_match"] = sum("error" not in r and all(r.get(f) == r["expected"][f] for f in ("production_date", "batch")) for r in records) / len(records)
    result["decision_accuracy"] = sum(r.get("decision") == r["expected"]["decision"] for r in records) / len(records)
    passed = [r for r in records if r.get("decision") == "PASS"]
    unsafe = [r for r in passed if r["expected"]["decision"] != "PASS" or any(r.get(f) != r["expected"][f] for f in ("production_date", "batch"))]
    result["pass_count"], result["wrong_pass_count"] = len(passed), len(unsafe)
    result["wrong_pass_rate_among_passes"] = len(unsafe) / len(passed) if passed else None
    total_chars = sum(r["reference_chars"] for r in records)
    result["cer"] = sum(r["edit_distance"] for r in records) / total_chars if total_chars else None
    times = [r["elapsed_ms"] for r in records if "elapsed_ms" in r]
    result["mean_ms"] = float(np.mean(times)) if times else None
    result["p95_ms"] = float(np.percentile(times, 95)) if times else None
    return result


def load_labels(path, split):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    groups, images = {}, set()
    for row in rows:
        for key in ("image", "group", "split", "source", "condition", "text", "production_date", "batch", "decision"):
            if key not in row:
                raise ValueError(f"标签缺少 {key}: {row}")
        if row["image"] in images:
            raise ValueError("标签包含重复图片")
        images.add(row["image"])
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("同一包装/合成源的变体跨集合，存在数据泄漏")
        if row["decision"] not in {"PASS", "REVIEW"}:
            raise ValueError("标签 decision 必须为 PASS 或 REVIEW")
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError(f"没有 {split} 样本")
    return rows, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="新的输出目录，拒绝覆盖")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--modes", nargs="+", choices=["raw", "enhance", "auto"], default=["raw", "auto"])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    all_rows, selected = load_labels(args.labels, args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    reader = Reader(args.threads)
    # 只预热一次，不计入逐图耗时；模型初始化单独记录。
    warmup = next((r for r in selected if (args.labels.parent / r["image"]).is_file()), None)
    if warmup:
        reader.recognize(read_image(args.labels.parent / warmup["image"]), mode="raw")
    report = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "dataset_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "split": args.split, "split_counts": dict(Counter(r["split"] for r in all_rows)),
        "source_counts": dict(Counter(r["source"] for r in selected)),
        "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("paddleocr", "paddlepaddle", "paddlex", "numpy", "opencv-contrib-python")},
        "cpu_threads": args.threads, "threshold": args.threshold, "model_init_ms": reader.init_ms,
        "timing_scope": "预热后，预处理+OCR+字段解析+坐标还原；不含图像解码、模型初始化、结果写盘；单轮运行",
        "model": {"det": "PP-OCRv5_mobile_det", "rec": "PP-OCRv5_mobile_rec", "device": "cpu", "enable_mkldnn": False},
        "modes": {},
    }
    for mode in dict.fromkeys(args.modes):
        records = []
        for i, label in enumerate(selected):
            record = {"image": label["image"], "condition": label["condition"], "expected": label}
            reference = normalized_text(label["text"])
            try:
                image = read_image(args.labels.parent / label["image"])
                result = reader.recognize(image, mode=mode, threshold=args.threshold, roi=label.get("roi"), angle=label.get("angle", 0))
                save_result(args.output / mode / f"{i:04d}", image, result)
                prediction = normalized_text("\n".join(r["text"] for r in result["rows"]))
                record.update({f: result["fields"][f]["value"] for f in ("production_date", "batch")})
                record.update(decision=result["decision"], elapsed_ms=result["elapsed_ms"], predicted_text=prediction)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                prediction = ""
            record.update(reference_chars=len(reference), edit_distance=edit_distance(reference, prediction))
            records.append(record)
            print(f"{mode} {i + 1}/{len(selected)} {label['image']}: {record.get('decision', record.get('error'))}", flush=True)
        report["modes"][mode] = {"overall": metrics(records), "by_condition": {c: metrics([r for r in records if r["condition"] == c]) for c in sorted({r["condition"] for r in records})}, "records": records}
        (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({m: r["overall"] for m, r in report["modes"].items()}, ensure_ascii=False, indent=2))
    return 1 if any(r["overall"]["errors"] for r in report["modes"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
