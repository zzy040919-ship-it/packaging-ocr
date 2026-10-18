"""图像预处理、PaddleOCR 3.x 推理、坐标还原和结果输出。"""
import json
import math
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from rules import DEFAULT_BATCH_PATTERN, extract_fields

ROOT = Path(__file__).resolve().parent
MAX_PIXELS = 24_000_000


def read_image(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("图像超过 2400 万像素，请先缩小或裁剪")
        return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


def rotate(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = math.ceil(h * sine + w * cosine), math.ceil(h * cosine + w * sine)
    matrix[:, 2] += [(nw - w) / 2, (nh - h) / 2]
    return cv2.warpAffine(image, matrix, (nw, nh), borderValue=(255, 255, 255)), matrix


def enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.addWeighted(gray, 1.4, cv2.GaussianBlur(gray, (0, 0), 1), -0.4, 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def group_rows(lines):
    """同一水平行中相邻的框合并。适用于已裁剪、单个包装的喷码区。"""
    groups = []
    for line in sorted(lines, key=lambda l: np.mean(np.asarray(l["box"])[:, 1])):
        box = np.asarray(line["box"])
        x0, y0 = box.min(axis=0)
        x1, y1 = box.max(axis=0)
        cy, height = (y0 + y1) / 2, max(y1 - y0, 1)
        target = next((g for g in groups if abs(g["cy"] - cy) < 0.45 * min(g["height"], height)), None)
        if target is None:
            target = dict(cy=cy, height=height, parts=[])
            groups.append(target)
        target["parts"].append((x0, x1, line))
    rows = []
    for group in groups:
        current, previous_x = [], None
        for x0, x1, line in sorted(group["parts"], key=lambda p: p[0]):
            if current and x0 - previous_x > 3 * group["height"]:
                rows.append({"text": " ".join(p["text"] for p in current), "confidence": min(p["confidence"] for p in current)})
                current = []
            current.append(line)
            previous_x = x1
        if current:
            rows.append({"text": " ".join(p["text"] for p in current), "confidence": min(p["confidence"] for p in current)})
    return rows


class Reader:
    def __init__(self, threads=4):
        if not 1 <= threads <= 64:
            raise ValueError("CPU 线程数必须在 1 到 64 之间")
        # 缓存放项目内，方便拷贝和离线运行。
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache" / "paddlex"))
        os.environ.setdefault("PADDLE_HOME", str(ROOT / ".cache" / "paddle"))
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".cache" / "modelscope"))
        # 显式传入已下载目录，断网时也能加载缓存。
        local_models = {}
        for kind, name in [("detection", "PP-OCRv5_mobile_det"), ("recognition", "PP-OCRv5_mobile_rec")]:
            directory = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models" / name
            if all((directory / f).is_file() for f in ("inference.json", "inference.pdiparams", "inference.yml")):
                local_models[f"text_{kind}_model_dir"] = str(directory)
        from paddleocr import PaddleOCR
        self.threads = threads
        start = perf_counter()
        self.engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu", cpu_threads=threads, enable_mkldnn=False,
            **local_models,
        )
        self.init_ms = round((perf_counter() - start) * 1000, 2)

    def _predict(self, image):
        results = list(self.engine.predict(image, text_rec_score_thresh=0.0))
        if len(results) != 1:
            raise RuntimeError("单张图片的 OCR 返回数量异常")
        res = results[0]
        texts, scores, polys = res["rec_texts"], res["rec_scores"], res["rec_polys"]
        if not (len(texts) == len(scores) == len(polys)):
            raise RuntimeError("OCR 返回的文本、置信度、坐标数量不一致")
        return [{"text": str(t), "confidence": round(float(s), 6), "box": np.asarray(p, dtype=float).tolist()}
                for t, s, p in zip(texts, scores, polys) if str(t).strip()]

    def recognize(self, image, *, mode="auto", roi=None, angle=0, threshold=0.8, batch_pattern=DEFAULT_BATCH_PATTERN):
        start = perf_counter()
        if mode not in {"raw", "enhance", "auto"}:
            raise ValueError("未知预处理模式")
        if not math.isfinite(angle) or not -180 <= angle <= 180:
            raise ValueError("旋转角度必须在 -180 到 180 之间")
        # 提前校验配置，即使图片没有识别到文字也报告配置错误。
        extract_fields([], threshold, batch_pattern)
        h, w = image.shape[:2]
        x, y, rw, rh = roi if roi is not None else (0, 0, w, h)
        if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
            raise ValueError("ROI 必须是图像内部的 x,y,width,height")
        crop = image[y:y + rh, x:x + rw]
        prepared, matrix = rotate(crop, angle)
        # ponytail: 限制最长边，极小喷码应使用 ROI，后续可按实测加入分块检测。
        scale = min(1.0, 1600 / max(prepared.shape[:2]))
        if scale < 1:
            prepared = cv2.resize(prepared, None, fx=scale, fy=scale)
        inverse = cv2.invertAffineTransform(matrix)

        def attempt(img, name, inverse_matrix):
            lines = self._predict(img)
            rows = group_rows(lines)
            result = extract_fields(rows, threshold, batch_pattern)
            for line in lines:
                points = np.asarray(line["box"], dtype=np.float32).reshape(1, -1, 2) / scale
                points = cv2.transform(points, inverse_matrix)[0] + np.array([x, y])
                line["box"] = points.round(2).tolist()
            result.update(lines=lines, rows=rows, variant=name)
            return result

        first = attempt(enhance(prepared) if mode == "enhance" else prepared, mode if mode != "auto" else "raw", inverse)
        attempts = [first]
        # 仅在字段未通过时重试；重试结果与首轮冲突时必须复核。
        if mode == "auto" and first["decision"] != "PASS":
            angles = []
            for line in first["lines"]:
                a, b = np.asarray(line["box"])[:2]
                degrees = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + angle
                if abs(degrees) <= 20:
                    angles.append(degrees)
            skew = float(np.median(angles)) if angles else 0.0
            retry, retry_matrix = rotate(prepared, skew if abs(skew) > 1 else 0)
            # 把重试坐标变回 prepared，然后变回原始图像。
            retry_inverse = cv2.invertAffineTransform(retry_matrix)
            combined = np.vstack([inverse, [0, 0, 1]]) @ np.diag([1 / scale, 1 / scale, 1]) @ np.vstack([retry_inverse, [0, 0, 1]]) @ np.diag([scale, scale, 1])
            attempts.append(attempt(enhance(retry), "enhance_deskew", combined[:2]))
        rank = lambda r: (sum(f["status"] == "VALID" for f in r["fields"].values()), sum(l["confidence"] for l in r["lines"]) / max(1, len(r["lines"])))
        chosen = max(attempts, key=rank)
        conflicts = []
        for key in chosen["fields"]:
            values = {c["value"] for a in attempts for c in a["fields"][key]["candidates"] if c["value"] is not None}
            if len(values) > 1:
                conflicts.append(key)
        if conflicts:
            chosen["decision"] = "REVIEW"
        chosen.update(
            conflicts=conflicts,
            attempts=[{"variant": a["variant"], "fields": a["fields"], "texts": [l["text"] for l in a["lines"]]} for a in attempts],
            elapsed_ms=round((perf_counter() - start) * 1000, 2),
            config={"mode": mode, "roi": roi, "angle": angle, "threshold": threshold, "batch_pattern": batch_pattern, "cpu_threads": self.threads},
            model={"det": "PP-OCRv5_mobile_det", "rec": "PP-OCRv5_mobile_rec", "device": "cpu"},
        )
        return chosen


def font(size=20):
    for path in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(image, result):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    label_font = font(max(14, min(28, image.shape[1] // 45)))
    for index, line in enumerate(result["lines"], 1):
        points = [tuple(p) for p in line["box"]]
        color = "#167454" if line["confidence"] >= result["config"]["threshold"] else "#c66a00"
        draw.line(points + [points[0]], fill=color, width=3)
        x, y = points[0]
        label = f"{index}: {line['text']} ({line['confidence']:.2f})"
        y = max(0, y - label_font.size - 5)
        draw.rectangle(draw.textbbox((x, y), label, font=label_font), fill="white")
        draw.text((x, y), label, fill=color, font=label_font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def save_result(directory, image, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_image(directory / "annotated.png", annotate(image, result))
