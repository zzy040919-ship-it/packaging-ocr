"""Tkinter 本地界面：选择图片、ROI、识别和导出。"""
from datetime import datetime
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from main import parse_roi
from pipeline import ROOT, Reader, annotate, read_image, save_result


class App:
    def __init__(self, root):
        self.root, self.reader = root, None
        self.image, self.result, self.path = None, None, None
        self.messages = queue.Queue()
        root.title("包装喷码识别 · PaddleOCR")
        root.geometry("1120x780")
        root.minsize(900, 680)
        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill="x")
        self.open_button = ttk.Button(toolbar, text="1. 选择图片", command=self.open_image)
        self.open_button.pack(side="left")
        self.run_button = ttk.Button(toolbar, text="2. 开始识别", command=self.run)
        self.run_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(toolbar, text="3. 导出结果", command=self.export, state="disabled")
        self.save_button.pack(side="left")
        self.mode = tk.StringVar(value="auto")
        ttk.Label(toolbar, text="预处理").pack(side="left", padx=(20, 5))
        ttk.Combobox(toolbar, textvariable=self.mode, values=["auto", "raw", "enhance"], state="readonly", width=9).pack(side="left")
        options = ttk.Frame(root, padding=(12, 0, 12, 8))
        options.pack(fill="x")
        self.roi, self.angle, self.threshold = tk.StringVar(), tk.StringVar(value="0"), tk.StringVar(value="0.8")
        for title, var, width in [("ROI x,y,w,h（留空为全图）", self.roi, 24), ("逆时针旋转°", self.angle, 6), ("置信度阈值", self.threshold, 6)]:
            ttk.Label(options, text=title).pack(side="left", padx=(0, 5))
            ttk.Entry(options, textvariable=var, width=width).pack(side="left", padx=(0, 18))
        self.status = tk.StringVar(value="选择一张包含生产日期、批次标签的图片。首次识别需要联网下载模型。")
        ttk.Label(root, textvariable=self.status, padding=(12, 5), wraplength=1060).pack(fill="x")
        self.canvas = ttk.Label(root, anchor="center")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.details = tk.Text(root, height=10, wrap="word", font=("Microsoft YaHei", 10))
        self.details.pack(fill="x", padx=12, pady=(0, 12))
        self.details.configure(state="disabled")
        root.after(100, self.poll)

    def show_image(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.thumbnail((1050, 390))
        self.preview = ImageTk.PhotoImage(image)
        self.canvas.configure(image=self.preview)

    def show_text(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)
        self.details.configure(state="disabled")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror("图片无法打开", str(exc))
            return
        self.image, self.path, self.result = image, path, None
        self.roi.set("")
        self.angle.set("0")
        self.save_button.configure(state="disabled")
        self.show_image(image)
        self.show_text("")
        self.status.set(f"{Path(path).name} · {image.shape[1]} × {image.shape[0]} 像素")

    def run(self):
        if self.image is None:
            messagebox.showinfo("尚未选择图片", "请先选择图片")
            return
        try:
            options = dict(mode=self.mode.get(), roi=parse_roi(self.roi.get()) if self.roi.get().strip() else None,
                           angle=float(self.angle.get()), threshold=float(self.threshold.get()))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.result = None
        for button in (self.open_button, self.run_button, self.save_button):
            button.configure(state="disabled")
        self.status.set("正在识别…首次运行包含模型加载与下载，请稍候。")
        self.show_text("")

        def work():
            try:
                if self.reader is None:
                    self.reader = Reader()
                result = self.reader.recognize(self.image, **options)
                result["input"] = self.path
                self.messages.put((result, None))
            except Exception as exc:
                self.messages.put((None, f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            result, error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.open_button.configure(state="normal")
            self.run_button.configure(state="normal")
            if error:
                self.status.set("识别失败，可检查网络和依赖后重试。")
                self.show_text(error)
            else:
                self.result = result
                self.save_button.configure(state="normal")
                self.show_image(annotate(self.image, result))
                decision = "通过规则检查" if result["decision"] == "PASS" else "需要人工复核"
                self.status.set(f"{decision} · {result['elapsed_ms']:.0f} ms · {result['variant']}（耗时不含模型加载）")
                lines = []
                for name, field in result["fields"].items():
                    title = "生产日期" if name == "production_date" else "批次号"
                    lines.append(f"{title}：{field['value'] or '—'}  [{field['status']}]  {field['reason']}")
                if result["conflicts"]:
                    lines.append("两轮识别存在冲突：" + ", ".join(result["conflicts"]))
                lines.append("\nOCR 原文：")
                lines.extend(f"{l['text']}    置信度 {l['confidence']:.3f}" for l in result["lines"])
                lines.append("\n通过仅表示文本符合配置规则；不能据此判断产品质量或喷码真实性。")
                self.show_text("\n".join(lines))
        self.root.after(100, self.poll)

    def export(self):
        if self.result is None:
            return
        directory = filedialog.askdirectory(title="选择导出父目录")
        if directory:
            target = Path(directory) / datetime.now().strftime("ocr_%Y%m%d_%H%M%S_%f")
            try:
                save_result(target, self.image, self.result)
                self.status.set(f"已导出：{target}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
