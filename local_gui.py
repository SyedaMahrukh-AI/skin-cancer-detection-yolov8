"""
Skin Cancer Detection System - Improved GUI
Uses YOLOv8 + ISIC Dataset (7 classes: Melanoma, Nevus, BCC, AK, BKL, DF, VASC)
Supports drag & drop via tkinterdnd2 — install with: pip install tkinterdnd2
"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\ANNLAB\Skin_cancer_detection 224,230,234\Semster Project\skin-cancer-project\models\best.pt"
WINDOW_TITLE = "Skin Cancer Detection System"
WINDOW_SIZE = "1100x720"

# Class colors for visualization (BGR for OpenCV, RGB for display)
CLASS_COLORS = {
    "Melanoma": "#ef4444",
    "Nevus":    "#22c55e",
    "BCC":      "#f59e0b",
    "AK":       "#8b5cf6",
    "BKL":      "#06b6d4",
    "DF":       "#f97316",
    "VASC":     "#ec4899",
}

# ─── DARK THEME COLORS ────────────────────────────────────────────────────────
BG_DARK      = "#0f172a"
BG_PANEL     = "#1e293b"
BG_CARD      = "#293548"
BG_INPUT     = "#334155"
ACCENT_BLUE  = "#3b82f6"
ACCENT_LIGHT = "#7dd3fc"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED   = "#94a3b8"
TEXT_DIM     = "#64748b"
BORDER       = "#334155"
SUCCESS      = "#22c55e"
WARNING      = "#f59e0b"
DANGER       = "#ef4444"


class SkinCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self.model = None
        self.selected_path = None
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.status_text = tk.StringVar(value="Loading model...")
        self.is_loading = False

        # Load model in background thread so GUI stays responsive
        threading.Thread(target=self._load_model, daemon=True).start()

        self._build_ui()

    # ── Model Loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
            self.root.after(0, lambda: self.status_text.set("✓ Model loaded — ready"))
            self.root.after(0, lambda: self.status_dot.config(fg=SUCCESS))
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set(f"✗ Model error: {e}"))
            self.root.after(0, lambda: self.status_dot.config(fg=DANGER))

    # ── UI Builder ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self.root, bg="#0a1628", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="⬡", font=("Courier", 22, "bold"),
            fg=ACCENT_LIGHT, bg="#0a1628"
        ).pack(side=tk.LEFT, padx=(18, 8), pady=10)

        tk.Label(
            header, text="SKIN CANCER DETECTION SYSTEM",
            font=("Courier", 13, "bold"), fg=TEXT_PRIMARY, bg="#0a1628"
        ).pack(side=tk.LEFT, pady=10)

        tk.Label(
            header, text="YOLOv8 · ISIC Dataset · 7 Classes",
            font=("Courier", 9), fg=TEXT_MUTED, bg="#0a1628"
        ).pack(side=tk.LEFT, padx=14, pady=10)

        # Status indicator (right side of header)
        status_frame = tk.Frame(header, bg="#0a1628")
        status_frame.pack(side=tk.RIGHT, padx=18)

        self.status_dot = tk.Label(
            status_frame, text="●", font=("Arial", 14),
            fg=WARNING, bg="#0a1628"
        )
        self.status_dot.pack(side=tk.LEFT)

        tk.Label(
            status_frame, textvariable=self.status_text,
            font=("Courier", 9), fg=TEXT_MUTED, bg="#0a1628"
        ).pack(side=tk.LEFT, padx=6)

        # ── Main layout ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Left panel — controls + image upload
        left = tk.Frame(main, bg=BG_PANEL, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Right panel — output image + results
        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_right_panel(right)

        # ── Bottom status bar ──
        bar = tk.Frame(self.root, bg="#0a1628", height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Label(
            bar,
            text="⚠  For research use only — not a substitute for clinical diagnosis",
            font=("Courier", 8), fg=TEXT_DIM, bg="#0a1628"
        ).pack(side=tk.LEFT, padx=12, pady=5)

    def _build_left_panel(self, parent):
        pad = {"padx": 14, "pady": 6}

        # Section label
        tk.Label(
            parent, text="INPUT IMAGE", font=("Courier", 9, "bold"),
            fg=TEXT_DIM, bg=BG_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(14, 4))

        # Upload zone (clickable)
        self.upload_zone = tk.Frame(
            parent, bg=BG_INPUT, height=170,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.upload_zone.pack(fill=tk.X, **pad)
        self.upload_zone.pack_propagate(False)
        self.upload_zone.bind("<Button-1>", lambda e: self.upload_image())
        self.upload_zone.bind("<Enter>", lambda e: self.upload_zone.config(bg=BG_CARD))
        self.upload_zone.bind("<Leave>", lambda e: self.upload_zone.config(bg=BG_INPUT))

        # Drag and drop support
        if DND_AVAILABLE:
            self.upload_zone.drop_target_register(DND_FILES)
            self.upload_zone.dnd_bind("<<Drop>>", self._on_drop)
            self.upload_zone.dnd_bind("<<DragEnter>>", lambda e: self.upload_zone.config(bg="#1e3a5f"))
            self.upload_zone.dnd_bind("<<DragLeave>>", lambda e: self.upload_zone.config(bg=BG_INPUT))

        self.upload_icon = tk.Label(
            self.upload_zone, text="↑", font=("Courier", 28, "bold"),
            fg=TEXT_DIM, bg=BG_INPUT
        )
        self.upload_icon.place(relx=0.5, rely=0.30, anchor=tk.CENTER)

        self.upload_label = tk.Label(
            self.upload_zone, text="Click or drag & drop image here",
            font=("Courier", 10), fg=TEXT_MUTED, bg=BG_INPUT
        )
        self.upload_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

        tk.Label(
            self.upload_zone, text="JPG · JPEG · PNG",
            font=("Courier", 8), fg=TEXT_DIM, bg=BG_INPUT
        ).place(relx=0.5, rely=0.72, anchor=tk.CENTER)

        # Preview thumbnail
        self.thumb_label = tk.Label(parent, bg=BG_PANEL)
        self.thumb_label.pack(**pad)

        # File name label
        self.file_label = tk.Label(
            parent, text="No file selected",
            font=("Courier", 8), fg=TEXT_DIM, bg=BG_PANEL,
            wraplength=280
        )
        self.file_label.pack(**pad)

        # Separator
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

        # Confidence threshold slider
        tk.Label(
            parent, text="CONFIDENCE THRESHOLD",
            font=("Courier", 9, "bold"), fg=TEXT_DIM, bg=BG_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(4, 2))

        slider_row = tk.Frame(parent, bg=BG_PANEL)
        slider_row.pack(fill=tk.X, padx=14)

        self.conf_slider = tk.Scale(
            slider_row, variable=self.conf_threshold,
            from_=0.05, to=0.95, resolution=0.05,
            orient=tk.HORIZONTAL, bg=BG_PANEL, fg=TEXT_PRIMARY,
            troughcolor=BG_INPUT, highlightthickness=0,
            activebackground=ACCENT_BLUE, showvalue=False,
            command=self._update_conf_label
        )
        self.conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.conf_label = tk.Label(
            slider_row, text="0.25",
            font=("Courier", 11, "bold"), fg=ACCENT_LIGHT, bg=BG_PANEL, width=4
        )
        self.conf_label.pack(side=tk.LEFT)

        # Separator
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

        # Buttons
        self.upload_btn = tk.Button(
            parent, text="  ▲  UPLOAD IMAGE",
            font=("Courier", 10, "bold"),
            bg=BG_CARD, fg=TEXT_PRIMARY,
            activebackground=BG_INPUT, activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT, bd=0, padx=0, pady=10,
            cursor="hand2", command=self.upload_image
        )
        self.upload_btn.pack(fill=tk.X, padx=14, pady=(0, 6))
        self._hover(self.upload_btn, BG_CARD, BG_INPUT)

        self.detect_btn = tk.Button(
            parent, text="  ◉  RUN DETECTION",
            font=("Courier", 10, "bold"),
            bg=ACCENT_BLUE, fg="white",
            activebackground="#2563eb", activeforeground="white",
            relief=tk.FLAT, bd=0, padx=0, pady=11,
            cursor="hand2", command=self.run_detection
        )
        self.detect_btn.pack(fill=tk.X, padx=14, pady=(0, 6))

        self.clear_btn = tk.Button(
            parent, text="  ✕  CLEAR",
            font=("Courier", 9),
            bg=BG_PANEL, fg=TEXT_DIM,
            activebackground=BG_INPUT, activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT, bd=0, padx=0, pady=8,
            cursor="hand2", command=self.clear_all
        )
        self.clear_btn.pack(fill=tk.X, padx=14)

    def _build_right_panel(self, parent):
        # Top row: input + output images
        img_row = tk.Frame(parent, bg=BG_DARK)
        img_row.pack(fill=tk.X, pady=(0, 10))

        # Input image display
        in_card = tk.Frame(img_row, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        in_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(
            in_card, text="INPUT", font=("Courier", 9, "bold"),
            fg=TEXT_DIM, bg=BG_PANEL
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.input_display = tk.Label(
            in_card, bg=BG_INPUT, width=350, height=280,
            text="Upload an image\nto begin",
            font=("Courier", 10), fg=TEXT_DIM
        )
        self.input_display.pack(padx=10, pady=(0, 10))

        # Output image display
        out_card = tk.Frame(img_row, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        out_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(
            out_card, text="DETECTION OUTPUT", font=("Courier", 9, "bold"),
            fg=TEXT_DIM, bg=BG_PANEL
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.output_display = tk.Label(
            out_card, bg=BG_INPUT, width=350, height=280,
            text="Annotated result\nappears here",
            font=("Courier", 10), fg=TEXT_DIM
        )
        self.output_display.pack(padx=10, pady=(0, 10))

        # Bottom: results panel
        results_frame = tk.Frame(parent, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        results_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            results_frame, text="DIAGNOSIS RESULTS",
            font=("Courier", 9, "bold"), fg=TEXT_DIM, bg=BG_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(10, 6))

        # Scrollable results area
        self.results_inner = tk.Frame(results_frame, bg=BG_PANEL)
        self.results_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        self._show_placeholder_results()

    # ── Helper: hover effect ────────────────────────────────────────────────
    def _hover(self, widget, normal_bg, hover_bg):
        widget.bind("<Enter>", lambda e: widget.config(bg=hover_bg))
        widget.bind("<Leave>", lambda e: widget.config(bg=normal_bg))

    def _update_conf_label(self, val):
        self.conf_label.config(text=f"{float(val):.2f}")

    # ── Placeholder results ─────────────────────────────────────────────────
    def _show_placeholder_results(self):
        for w in self.results_inner.winfo_children():
            w.destroy()

        tk.Label(
            self.results_inner,
            text="Run detection to see results here",
            font=("Courier", 10), fg=TEXT_DIM, bg=BG_PANEL
        ).pack(pady=10)

        # Class legend
        legend = tk.Frame(self.results_inner, bg=BG_PANEL)
        legend.pack(fill=tk.X, pady=4)

        for cls, color in CLASS_COLORS.items():
            chip = tk.Frame(legend, bg=BG_PANEL)
            chip.pack(side=tk.LEFT, padx=3)
            tk.Label(chip, text="●", fg=color, bg=BG_PANEL, font=("Arial", 8)).pack(side=tk.LEFT)
            tk.Label(chip, text=cls, font=("Courier", 8), fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT)

    def _show_results(self, predictions):
        for w in self.results_inner.winfo_children():
            w.destroy()

        if not predictions:
            tk.Label(
                self.results_inner,
                text="No lesion detected — try lowering the confidence threshold",
                font=("Courier", 10), fg=WARNING, bg=BG_PANEL
            ).pack(pady=10)
            return

        tk.Label(
            self.results_inner,
            text=f"{len(predictions)} detection(s) found",
            font=("Courier", 9), fg=TEXT_MUTED, bg=BG_PANEL
        ).pack(anchor=tk.W, pady=(0, 6))

        for name, conf in predictions:
            color = CLASS_COLORS.get(name, ACCENT_BLUE)
            conf_pct = int(conf * 100)

            row = tk.Frame(self.results_inner, bg=BG_CARD, pady=8)
            row.pack(fill=tk.X, pady=3)

            # Color dot
            tk.Label(row, text="  ●", fg=color, bg=BG_CARD, font=("Arial", 14)).pack(side=tk.LEFT)

            # Class name
            tk.Label(
                row, text=name,
                font=("Courier", 12, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD
            ).pack(side=tk.LEFT, padx=8)

            # Confidence bar
            bar_frame = tk.Frame(row, bg=BG_CARD)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

            bar_bg = tk.Frame(bar_frame, bg=BG_INPUT, height=8)
            bar_bg.pack(fill=tk.X, pady=6)
            bar_bg.update_idletasks()

            bar_fill = tk.Frame(bar_bg, bg=color, height=8)
            bar_fill.place(x=0, y=0, relwidth=conf, relheight=1)

            # Confidence %
            conf_color = SUCCESS if conf >= 0.7 else (WARNING if conf >= 0.5 else DANGER)
            tk.Label(
                row, text=f"{conf_pct}%",
                font=("Courier", 11, "bold"), fg=conf_color, bg=BG_CARD
            ).pack(side=tk.RIGHT, padx=10)

    # ── Actions ─────────────────────────────────────────────────────────────
    def run_detection(self):
        if not self.selected_path:
            self.status_text.set("Please upload an image first")
            return
        if self.model is None:
            self.status_text.set("Model is still loading — please wait")
            return
        if self.is_loading:
            return

        self.is_loading = True
        self.detect_btn.config(text="  ◌  DETECTING...", state=tk.DISABLED, bg="#1d4ed8")
        self.status_text.set("Running detection...")
        self.status_dot.config(fg=WARNING)

        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        try:
            results = self.model.predict(
                source=self.selected_path,
                conf=self.conf_threshold.get()
            )
            result = results[0]

            # Annotated output image
            annotated = result.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_out = Image.fromarray(annotated_rgb).resize((330, 260))
            photo_out = ImageTk.PhotoImage(img_out)

            # Predictions list
            predictions = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = self.model.names[cls]
                    conf = float(box.conf[0])
                    predictions.append((name, conf))

            self.root.after(0, lambda: self._update_output(photo_out, predictions))

        except Exception as e:
            self.root.after(0, lambda: self.status_text.set(f"Error: {e}"))
            self.root.after(0, lambda: self.status_dot.config(fg=DANGER))
        finally:
            self.is_loading = False
            self.root.after(0, lambda: self.detect_btn.config(
                text="  ◉  RUN DETECTION", state=tk.NORMAL, bg=ACCENT_BLUE
            ))

    def _update_output(self, photo_out, predictions):
        self.output_display.config(image=photo_out, text="")
        self.output_display.image = photo_out
        self._show_results(predictions)
        count = len(predictions)
        self.status_text.set(f"Done — {count} detection(s) found")
        self.status_dot.config(fg=SUCCESS)

    def clear_all(self):
        self.selected_path = None
        self.input_display.config(image="", text="Upload an image\nto begin")
        self.output_display.config(image="", text="Annotated result\nappears here")
        self.upload_icon.config(image="", text="↑")
        self.upload_label.config(text="Click or drag & drop image here")
        self.file_label.config(text="No file selected", fg=TEXT_DIM)
        self.status_text.set("Cleared — ready")
        self.status_dot.config(fg=SUCCESS)
        self._show_placeholder_results()

    def _on_drop(self, event):
        # tkinterdnd2 gives path wrapped in {} on Windows, strip it
        path = event.data.strip().strip("{}")
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            self.selected_path = path
            self._load_image_preview(path)
        else:
            self.status_text.set("Unsupported file — use JPG or PNG")
            self.status_dot.config(fg=DANGER)

    def _load_image_preview(self, path):
        fname = os.path.basename(path)
        self.file_label.config(text=fname, fg=TEXT_MUTED)

        # Thumbnail in upload zone
        img = Image.open(path).resize((120, 90))
        photo = ImageTk.PhotoImage(img)
        self.upload_icon.config(image=photo, text="")
        self.upload_icon.image = photo
        self.upload_label.config(text=fname[:28] + "..." if len(fname) > 28 else fname)

        # Full image in input display
        img_full = Image.open(path).resize((330, 260))
        photo_full = ImageTk.PhotoImage(img_full)
        self.input_display.config(image=photo_full, text="")
        self.input_display.image = photo_full

        self.status_text.set("Image loaded — click Run Detection")
        self.status_dot.config(fg=ACCENT_LIGHT)
        self.upload_zone.config(bg=BG_INPUT)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select skin lesion image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return
        self.selected_path = path
        self._load_image_preview(path)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("Drag & drop not available. Install with: pip install tkinterdnd2")
    app = SkinCancerApp(root)
    root.mainloop()
