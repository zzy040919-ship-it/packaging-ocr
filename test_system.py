"""无需下载 OCR 模型的回归测试：规则边界、坐标还原、重试冲突、评估分母。"""
import tempfile
import json
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import cv2

from evaluate import edit_distance, load_labels, metrics, warm_up
from pipeline import Reader, group_rows, read_image, rotate, save_image, select_device
from rules import extract_fields


def rows(*texts, confidence=0.99):
    return [{"text": t, "confidence": confidence} for t in texts]


class SystemTests(unittest.TestCase):
    @unittest.skipIf(os.name != "nt" and not os.environ.get("DISPLAY"), "Tk GUI 检查需要显示服务器")
    def test_gui_runtime_options_and_close_guard(self):
        import tkinter as tk
        from gui import App
        root = tk.Tk()
        root.withdraw()
        result = extract_fields(rows("MFG:20261018", "LOT:AB1234"))
        result.update(lines=[], conflicts=[], elapsed_ms=1, variant="raw", model={"device": "cpu"},
                      config={"threshold": .8, "runtime": {"effective_device": "cpu"}})
        reader = Mock()
        reader.recognize.return_value = result
        try:
            app = App(root)
            app.image, app.path = np.zeros((80, 200, 3), dtype=np.uint8), "synthetic-gui-test.png"
            app.device.set("cpu")
            app.threads.set("2")
            app.max_side.set("640")
            with patch("gui.Reader", return_value=reader) as factory:
                app.run()
                app.close()
                self.assertTrue(root.winfo_exists())
                deadline = time.monotonic() + 5
                while app.busy and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertFalse(app.busy)
                self.assertEqual(app.result["decision"], "PASS")
                factory.assert_called_once_with(device="cpu", threads=2, enable_mkldnn=False, max_side=640)
                self.assertEqual(str(app.save_button["state"]), "normal")
        finally:
            root.destroy()

    def test_partial_dates_spacing_and_nonfinite_confidence(self):
        for date_text in ["2026.10.18.99", "2026-10/18", "2026.10.180"]:
            with self.subTest(date_text=date_text):
                self.assertEqual(extract_fields(rows("MFG:" + date_text, "LOT:AB1234"))["decision"], "REVIEW")
        for date_text in ["2026 . 10 . 18", "2026 年 10 月 18 日", "2026/10/18"]:
            self.assertEqual(extract_fields(rows("MFG:" + date_text, "LOT:AB1234"))["decision"], "PASS")
        for confidence in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                extract_fields(rows("MFG:20261018", "LOT:AB1234", confidence=confidence))

    def test_runtime_device_selection(self):
        paddle = SimpleNamespace(device=SimpleNamespace(is_compiled_with_cuda=lambda: False, cuda=SimpleNamespace(device_count=lambda: 0)))
        self.assertEqual(select_device("auto", paddle)[0], "cpu")
        self.assertIsNotNone(select_device("auto", paddle)[1])
        with self.assertRaises(ValueError):
            select_device("gpu:0", paddle)
        paddle.device.is_compiled_with_cuda = lambda: True
        paddle.device.cuda.device_count = lambda: 2
        self.assertEqual(select_device("auto", paddle), ("gpu:0", None))
        self.assertEqual(select_device("gpu:1", paddle), ("gpu:1", None))
        with self.assertRaises(ValueError):
            select_device("gpu:2", paddle)
        with self.assertRaises(ValueError):
            select_device("made-up-device", paddle)

    def test_runtime_constructor_and_auto_fallback(self):
        fake_paddle = SimpleNamespace(device=SimpleNamespace(is_compiled_with_cuda=lambda: True, cuda=SimpleNamespace(device_count=lambda: 1)))
        factory = Mock(side_effect=[RuntimeError("test GPU init failure"), Mock()])
        with patch.dict("sys.modules", {"paddle": fake_paddle, "paddleocr": SimpleNamespace(PaddleOCR=factory)}), patch("pipeline.os.cpu_count", return_value=2):
            reader = Reader(device="auto", max_side=960)
            self.assertEqual(reader.threads, 2)
            self.assertEqual(reader.runtime_info["effective_device"], "cpu")
            self.assertIn("test GPU init failure", reader.runtime_info["fallback_reason"])
            self.assertEqual(factory.call_args.kwargs["device"], "cpu")
            factory.reset_mock(side_effect=True)
            factory.side_effect = RuntimeError("force GPU failure")
            with self.assertRaises(RuntimeError):
                Reader(device="gpu:0")
            self.assertEqual(factory.call_count, 1)

    def test_invalid_ocr_output_rejected(self):
        reader = Reader.__new__(Reader)
        reader.engine = Mock()
        for score, poly in [(float("nan"), np.zeros((4, 2))), (.99, np.zeros((3, 2))), (.99, np.full((4, 2), float("inf")))]:
            reader.engine.predict.return_value = [{"rec_texts": ["MFG:20261018"], "rec_scores": [score], "rec_polys": [poly]}]
            with self.assertRaises(RuntimeError):
                reader._predict(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_corrupt_warmup_and_empty_metrics(self):
        reader = Mock()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "bad.png").write_bytes(b"not an image")
            save_image(folder / "good.png", np.zeros((20, 20, 3), dtype=np.uint8))
            result = warm_up(reader, [{"image": "bad.png"}, {"image": "good.png"}], folder)
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["warnings"]), 1)
            self.assertEqual(result["image"], "good.png")
        self.assertIsNone(metrics([])["both_fields_exact_match"])

    def test_dataset_alias_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            row = dict(image="images/a.png", group="A", split="test", source="synthetic", condition="test", text="", production_date=None, batch=None, decision="REVIEW")
            path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "image": "images/../images/a.png", "group": "B"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labels(path, "test")

    def test_calendar_labels_and_confidence(self):
        result = extract_fields(rows("MFG:2024.02.29", "LOT:A240229B01"))
        self.assertEqual(result["decision"], "PASS")
        for text, status in [("MFG:2025.02.29", "INVALID"), ("20261018", "UNCONFIRMED"), ("EXP:2027.10.18", "MISSING")]:
            with self.subTest(text=text):
                self.assertEqual(extract_fields(rows(text))["fields"]["production_date"]["status"], status)
        result = extract_fields(rows("生产日期：２０２６年１０月１８日", "批号：AB1234", confidence=0.5))
        self.assertEqual(result["fields"]["production_date"]["status"], "LOW_CONFIDENCE")
        self.assertEqual(result["decision"], "REVIEW")
        result = extract_fields(rows("MFG:20261018 EXP:20271018", "LOT:AB1234"))
        self.assertEqual(result["fields"]["production_date"]["value"], "2026-10-18")
        result = extract_fields(rows("MFG:20261018 LOT:20261019"))
        self.assertEqual(result["fields"]["production_date"]["value"], "2026-10-18")
        self.assertEqual(result["fields"]["batch"]["value"], "20261019")

    def test_ambiguity_and_no_silent_repair(self):
        result = extract_fields(rows("MFG:20261018", "MFG:20261019", "LOT:AB1234"))
        self.assertEqual(result["fields"]["production_date"]["status"], "AMBIGUOUS")
        result = extract_fields(rows("MFG:2O26.10.18", "LOT:AB@1234"))
        self.assertEqual(result["decision"], "REVIEW")
        self.assertEqual(result["fields"]["batch"]["status"], "INVALID")
        with self.assertRaises(ValueError):
            extract_fields([], threshold=1.5)

    def test_geometry_and_chinese_path(self):
        image = np.full((300, 900, 3), 245, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文图片.png"
            save_image(path, image)
            self.assertTrue(np.array_equal(read_image(path), image))
        reader = Reader.__new__(Reader)
        reader.threads = 4
        reader._predict = lambda img: [{"text": "MFG:20261018", "confidence": 0.99, "box": [[0, 0], [60, 0], [60, 20], [0, 20]]}]
        result = reader.recognize(image, mode="raw", roi=(100, 50, 300, 150))
        np.testing.assert_allclose(result["lines"][0]["box"][0], [100, 50])
        with self.assertRaises(ValueError):
            reader.recognize(image, roi=(900, 0, 10, 10))

    def test_conflicting_retry(self):
        reader = Reader.__new__(Reader)
        reader.threads = 4
        predictions = iter([
            [{"text": "MFG:20261018", "confidence": 0.7, "box": [[10, 10], [160, 10], [160, 30], [10, 30]]}],
            [{"text": "MFG:20261019 LOT:AB1234", "confidence": 0.99, "box": [[10, 10], [260, 10], [260, 30], [10, 30]]}],
        ])
        reader._predict = lambda img: next(predictions)
        result = reader.recognize(np.full((100, 400, 3), 255, dtype=np.uint8))
        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("production_date", result["conflicts"])

    def test_deskew_retry_restores_original_coordinates(self):
        reader = Reader.__new__(Reader)
        reader.threads = 4
        image = np.full((1800, 2800, 3), 255, dtype=np.uint8)
        points = np.array([[100, 100], [800, 240], [790, 290], [90, 150]], dtype=np.float32)
        angle = float(np.degrees(np.arctan2(140, 700)))
        prepared = cv2.resize(image, None, fx=1600 / 2800, fy=1600 / 2800)
        _, matrix = rotate(prepared, angle)
        scaled = points * (1600 / 2800)
        second = cv2.transform(scaled.reshape(1, -1, 2), matrix)[0]
        predictions = iter([
            [{"text": "MFG:20261018", "confidence": 0.7, "box": scaled.tolist()}],
            [{"text": "MFG:20261018 LOT:AB1234", "confidence": 0.99, "box": second.tolist()}],
        ])
        reader._predict = lambda img: next(predictions)
        result = reader.recognize(image)
        self.assertEqual(result["variant"], "enhance_deskew")
        np.testing.assert_allclose(result["lines"][0]["box"], points, atol=0.1)

    def test_ambiguous_first_attempt_cannot_be_overridden(self):
        reader = Reader.__new__(Reader)
        reader.threads = 4
        def line(text, y):
            return {"text": text, "confidence": 0.99, "box": [[10, y], [300, y], [300, y + 20], [10, y + 20]]}
        predictions = iter([
            [line("MFG:20261018", 10), line("MFG:20261019 LOT:AB1234", 50)],
            [line("MFG:20261018 LOT:AB1234", 10)],
        ])
        reader._predict = lambda img: next(predictions)
        result = reader.recognize(np.full((100, 400, 3), 255, dtype=np.uint8))
        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("production_date", result["conflicts"])

    def test_metrics_include_errors(self):
        reference = {"production_date": "2026-10-18", "batch": "AB1234", "decision": "PASS"}
        records = [{"expected": reference, "production_date": "2026-10-18", "batch": "AB1234", "decision": "PASS", "reference_chars": 5, "edit_distance": 0, "elapsed_ms": 10},
                   {"expected": reference, "error": "bad file", "reference_chars": 5, "edit_distance": 5}]
        report = metrics(records)
        self.assertEqual(report["production_date_exact_match"], 0.5)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["cer"], 0.5)
        self.assertEqual(edit_distance("ABC", "AXBC"), 1)


if __name__ == "__main__":
    unittest.main()
