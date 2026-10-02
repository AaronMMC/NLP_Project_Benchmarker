"""
gui.py — Simple desktop GUI for the Ilocano Hymn Lexicon Benchmark.

Run it:
    py -3.13 gui.py

What it does:
  - Run      : execute the full benchmark (fast or full mode) with live log output
  - Report   : view results/REPORT.md
  - Figures  : browse all generated PNG plots (with open-file / open-folder)
  - Metrics  : browse all CSV metric tables
  - Search   : interactive song retrieval ("which song is this phrase from?")
  - Logs     : view the run log, corpus inspection, normalisation rules

Everything the CLI produces is reachable here. The CLI itself remains:
    py -3.13 -m src.run_benchmark --config config.yaml
    py -3.13 -m src.retrieval "Apo caasiannacami"
"""

from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — GUI always works from the project root (folder that contains this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
ERRORS_DIR = RESULTS_DIR / "errors"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Python launcher used for subprocesses (same interpreter that started the GUI)
PYTHON = sys.executable

try:
    from PIL import Image, ImageTk  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext


# =============================================================================
# Helpers
# =============================================================================
def open_path(path: Path | str) -> None:
    """Open a file or folder with the OS default handler (Windows: os.startfile)."""
    path = Path(path)
    if not path.exists():
        messagebox.showinfo("Not found", f"Does not exist yet:\n{path}\n\nRun the benchmark first.")
        return
    try:
        if path.is_dir():
            os.startfile(str(path))  # noqa: S606
        else:
            os.startfile(str(path.resolve()))  # noqa: S606
    except Exception as exc:  # pragma: no cover - OS specific
        messagebox.showerror("Could not open", f"{path}\n\n{exc}")


def list_sorted(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Return existing files in directory matching any of the glob patterns, sorted by name."""
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    # de-duplicate while keeping sort order
    return sorted({f.resolve() for f in files if f.is_file()}, key=lambda p: p.name.lower())


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Could not read {path.name}: {exc}]"


def python_version_line() -> str:
    return f"Python {sys.version.split()[0]}  |  {PYTHON}"


# =============================================================================
# Main application
# =============================================================================
class BenchmarkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ilocano Hymn Lexicon Benchmark")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self._proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._retriever = None
        self._retriever_status = tk.StringVar(value="Index not loaded yet (click Load index)")
        self._image_refs: list[ImageTk.PhotoImage] = []  # prevent GC of displayed images

        self._build_ui()
        self._poll_log_queue()
        self.after(400, self._refresh_all_browsers)

        # Status bar
        self.status_var = tk.StringVar(value=f"Ready  |  {python_version_line()}")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except Exception:
            try:
                style.theme_use("clam")
            except Exception:
                pass

        top = ttk.Frame(self, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Ilocano Hymn Lexicon Benchmark", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Open project folder", command=lambda: open_path(PROJECT_ROOT)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Refresh outputs", command=self._refresh_all_browsers).pack(side=tk.RIGHT, padx=4)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 4))

        self._build_run_tab()
        self._build_report_tab()
        self._build_figures_tab()
        self._build_metrics_tab()
        self._build_search_tab()
        self._build_logs_tab()

    # ---- Tab 1: Run -----------------------------------------------------
    def _build_run_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Run benchmark ")

        opts = ttk.Frame(tab)
        opts.pack(fill=tk.X, pady=(0, 6))

        self.fast_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts,
            text="Fast mode (fewer folds/repeats/bootstraps — quicker, less stable numbers)",
            variable=self.fast_var,
        ).pack(side=tk.LEFT)

        ttk.Button(opts, text="▶  Run benchmark", command=self._run_benchmark).pack(side=tk.RIGHT)
        self.stop_btn = ttk.Button(opts, text="■  Stop", command=self._stop_benchmark, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=6)

        # Progress / state
        self.run_state_var = tk.StringVar(value="Idle")
        ttk.Label(tab, textvariable=self.run_state_var, foreground="#333").pack(anchor=tk.W, pady=(0, 4))

        # Live console
        console_frame = ttk.LabelFrame(tab, text="Live output (benchmark log)", padding=4)
        console_frame.pack(expand=True, fill=tk.BOTH)

        self.console = scrolledtext.ScrolledText(
            console_frame, wrap=tk.WORD, height=24, font=("Consolas", 9), state=tk.DISABLED
        )
        self.console.pack(expand=True, fill=tk.BOTH)
        self.console.tag_configure("err", foreground="#b00020")
        self.console.tag_configure("ok", foreground="#0a6b2d")

        # Output shortcuts
        out = ttk.Frame(tab)
        out.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(out, text="Outputs:").pack(side=tk.LEFT)
        for label, target in [
            ("REPORT.md", REPORT_PATH),
            ("metrics/", METRICS_DIR),
            ("figures/", FIGURES_DIR),
            ("errors/", ERRORS_DIR),
            ("logs/", LOGS_DIR),
        ]:
            ttk.Button(out, text=label, command=lambda t=target: open_path(t)).pack(side=tk.LEFT, padx=3)

    # ---- Tab 2: Report --------------------------------------------------
    def _build_report_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Report ")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(bar, text="Reload REPORT.md", command=self._load_report).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open file", command=lambda: open_path(REPORT_PATH)).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text=str(REPORT_PATH), foreground="#666").pack(side=tk.LEFT, padx=8)

        self.report_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=("Segoe UI", 10))
        self.report_text.pack(expand=True, fill=tk.BOTH)

    # ---- Tab 3: Figures -------------------------------------------------
    def _build_figures_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Figures ")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(bar, text="Figure:").pack(side=tk.LEFT)
        self.figure_var = tk.StringVar()
        self.figure_combo = ttk.Combobox(bar, textvariable=self.figure_var, state="readonly", width=45)
        self.figure_combo.pack(side=tk.LEFT, padx=6)
        self.figure_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_figure())
        ttk.Button(bar, text="Open image file", command=self._open_selected_figure).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Open figures folder", command=lambda: open_path(FIGURES_DIR)).pack(side=tk.LEFT, padx=4)

        self.fig_preview = ttk.Label(tab, text="Select a figure (run the benchmark first if the list is empty).", anchor=tk.CENTER)
        self.fig_preview.pack(expand=True, fill=tk.BOTH, pady=4)

    # ---- Tab 4: Metrics -------------------------------------------------
    def _build_metrics_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Metrics ")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(bar, text="CSV:").pack(side=tk.LEFT)
        self.metric_var = tk.StringVar()
        self.metric_combo = ttk.Combobox(bar, textvariable=self.metric_var, state="readonly", width=45)
        self.metric_combo.pack(side=tk.LEFT, padx=6)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_metric())
        ttk.Button(bar, text="Open CSV file", command=self._open_selected_metric).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Open metrics folder", command=lambda: open_path(METRICS_DIR)).pack(side=tk.LEFT, padx=4)

        # Treeview table
        table_frame = ttk.Frame(tab)
        table_frame.pack(expand=True, fill=tk.BOTH, pady=4)
        self.metric_tree = ttk.Treeview(table_frame, show="headings", height=20)
        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.metric_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.metric_tree.xview)
        self.metric_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.metric_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.metric_raw = scrolledtext.ScrolledText(tab, height=8, font=("Consolas", 9))
        self.metric_raw.pack(fill=tk.X, pady=(4, 0))

    # ---- Tab 5: Song search ---------------------------------------------
    def _build_search_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Song search ")

        info = ttk.Label(
            tab,
            text="Type a phrase from a hymn/song and get the top matching songs (Task C retrieval).",
        )
        info.pack(anchor=tk.W, pady=(0, 6))

        qbar = ttk.Frame(tab)
        qbar.pack(fill=tk.X)
        self.query_var = tk.StringVar()
        entry = ttk.Entry(qbar, textvariable=self.query_var)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        entry.bind("<Return>", lambda _e: self._do_search())
        ttk.Button(qbar, text="Search", command=self._do_search).pack(side=tk.LEFT, padx=6)
        ttk.Button(qbar, text="Load index", command=self._load_retriever).pack(side=tk.LEFT)
        ttk.Label(qbar, text="Top-k:").pack(side=tk.LEFT, padx=(8, 2))
        self.topk_var = tk.StringVar(value="5")
        ttk.Combobox(qbar, textvariable=self.topk_var, values=["1", "3", "5", "10"], width=4, state="readonly").pack(side=tk.LEFT)

        ttk.Label(tab, textvariable=self._retriever_status, foreground="#444").pack(anchor=tk.W, pady=4)

        results_frame = ttk.LabelFrame(tab, text="Results", padding=4)
        results_frame.pack(expand=True, fill=tk.BOTH)
        self.search_tree = ttk.Treeview(
            results_frame, columns=("rank", "song", "title", "score"), show="headings", height=16
        )
        self.search_tree.heading("rank", text="#")
        self.search_tree.heading("song", text="Song ID")
        self.search_tree.heading("title", text="Title")
        self.search_tree.heading("score", text="Score")
        self.search_tree.column("rank", width=40, anchor=tk.CENTER)
        self.search_tree.column("song", width=80, anchor=tk.CENTER)
        self.search_tree.column("title", width=520)
        self.search_tree.column("score", width=100, anchor=tk.E)
        vsb = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=vsb.set)
        self.search_tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Example queries
        ex = ttk.LabelFrame(tab, text="Example queries (click to fill)", padding=6)
        ex.pack(fill=tk.X, pady=(8, 0))
        for phrase in [
            "Apo caasiannacami",
            "Gloria coma iti Ama",
            "Santo, Santo, Santo",
        ]:
            ttk.Button(ex, text=phrase, command=lambda p=phrase: self._fill_query(p)).pack(side=tk.LEFT, padx=3)

    # ---- Tab 6: Logs ----------------------------------------------------
    def _build_logs_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Logs ")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 6))
        self.log_file_var = tk.StringVar()
        self.log_combo = ttk.Combobox(bar, textvariable=self.log_file_var, state="readonly", width=45)
        self.log_combo.pack(side=tk.LEFT, padx=6)
        self.log_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_log())
        ttk.Button(bar, text="Reload", command=self._show_selected_log).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Open file", command=self._open_selected_log).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Open logs folder", command=lambda: open_path(LOGS_DIR)).pack(side=tk.LEFT, padx=4)

        self.log_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(expand=True, fill=tk.BOTH)

    # ------------------------------------------------------------ Run logic
    def _append_console(self, text: str, tag: str | None = None) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, text, tag or ())
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _run_benchmark(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            messagebox.showinfo("Already running", "A benchmark run is already in progress.")
            return
        if not CONFIG_PATH.exists():
            messagebox.showerror("Missing config", f"config.yaml not found at:\n{CONFIG_PATH}")
            return

        cmd = [PYTHON, "-m", "src.run_benchmark", "--config", str(CONFIG_PATH)]
        if self.fast_var.get():
            cmd.append("--fast")

        self._append_console("\n" + "=" * 70 + "\n", "ok")
        self._append_console(f"$ {' '.join(cmd)}\n", "ok")
        self.run_state_var.set("Running… (full mode can take several minutes)")
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Benchmark running…")

        # Stream stdout+stderr on a background thread into a queue
        def _reader(proc: subprocess.Popen) -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._log_queue.put(line)
                for line in proc.stderr:
                    self._log_queue.put(line)
            except Exception as exc:  # pragma: no cover
                self._log_queue.put(f"[reader error] {exc}\n")
            finally:
                self._log_queue.put("__PROCESS_END__\n")

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            self.run_state_var.set("Failed to start")
            self.stop_btn.configure(state=tk.DISABLED)
            messagebox.showerror("Launch failed", str(exc))
            return

        self._reader_thread = threading.Thread(target=_reader, args=(self._proc,), daemon=True)
        self._reader_thread.start()

    def _poll_log_queue(self) -> None:
        """Drain the subprocess log queue into the console (runs on the Tk thread)."""
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line == "__PROCESS_END__\n":
                    self._on_process_end()
                else:
                    tag = "err" if ("ERROR" in line or "Traceback" in line or "failed" in line.lower()) else None
                    self._append_console(line, tag)
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _on_process_end(self) -> None:
        rc = self._proc.returncode if self._proc else None
        self._proc = None
        self.stop_btn.configure(state=tk.DISABLED)
        if rc == 0:
            self.run_state_var.set("Completed successfully.")
            self._append_console("\n✅ Benchmark finished. Outputs refreshed below.\n", "ok")
            self.status_var.set("Benchmark complete")
        elif rc is None:
            self.run_state_var.set("Stopped by user.")
        else:
            self.run_state_var.set(f"Exited with code {rc}. Check the log above.")
            self.status_var.set(f"Benchmark failed (code {rc})")
        self._refresh_all_browsers()

    def _stop_benchmark(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self.run_state_var.set("Terminating…")
        except Exception as exc:
            messagebox.showerror("Stop failed", str(exc))

    # ------------------------------------------------------------ Browsers
    def _refresh_all_browsers(self) -> None:
        self._refresh_figures()
        self._refresh_metrics()
        self._refresh_logs()
        if REPORT_PATH.exists():
            self._load_report()
        else:
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(
                tk.END,
                "REPORT.md has not been generated yet.\n\n"
                "Click 'Run benchmark' on the Run tab, or from a terminal:\n"
                f"    {PYTHON} -m src.run_benchmark --config config.yaml\n",
            )

    def _load_report(self) -> None:
        self.report_text.delete("1.0", tk.END)
        if REPORT_PATH.exists():
            self.report_text.insert(tk.END, read_text(REPORT_PATH))
        else:
            self.report_text.insert(tk.END, "REPORT.md not found. Run the benchmark first.")

    # Figures
    def _refresh_figures(self) -> None:
        files = list_sorted(FIGURES_DIR, ("*.png", "*.svg", "*.jpg", "*.jpeg"))
        names = [f.name for f in files]
        self.figure_combo["values"] = names
        if names and self.figure_var.get() not in names:
            self.figure_var.set(names[0])
        self._current_figures = files
        if names:
            self._show_selected_figure()
        else:
            self.fig_preview.configure(image="", text="No figures yet — run the benchmark first.")
            self._image_refs.clear()

    def _selected_figure(self) -> Path | None:
        files = getattr(self, "_current_figures", [])
        name = self.figure_var.get()
        for f in files:
            if f.name == name:
                return f
        return None

    def _show_selected_figure(self) -> None:
        path = self._selected_figure()
        if path is None:
            self.fig_preview.configure(image="", text="No figure selected.")
            self._image_refs.clear()
            return
        if path.suffix.lower() != ".png" or not HAS_PIL:
            # SVG / no PIL — just show the path
            self.fig_preview.configure(image="", text=f"{path.name}\n\n{path}\n\n(Open the file to view it.)")
            self._image_refs.clear()
            return
        try:
            img = Image.open(path)
            # Fit into a generous preview area
            max_w, max_h = 1000, 560
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._image_refs = [photo]  # keep reference
            self.fig_preview.configure(image=photo, text="")
        except Exception as exc:
            self.fig_preview.configure(image="", text=f"Could not render {path.name}: {exc}")

    def _open_selected_figure(self) -> None:
        path = self._selected_figure()
        if path is None:
            messagebox.showinfo("No figure", "Select a figure first.")
            return
        open_path(path)

    # Metrics
    def _refresh_metrics(self) -> None:
        files = list_sorted(METRICS_DIR, ("*.csv", "*.tsv", "*.json", "*.md"))
        names = [f.name for f in files]
        self.metric_combo["values"] = names
        if names and self.metric_var.get() not in names:
            # Prefer a headline table if present
            preferred = next((n for n in names if "task_c" in n.lower() or "coverage" in n.lower()), names[0])
            self.metric_var.set(preferred)
        self._current_metrics = files
        if names:
            self._show_selected_metric()
        else:
            self._clear_metric_table()
            self.metric_raw.delete("1.0", tk.END)
            self.metric_raw.insert(tk.END, "No metric files yet — run the benchmark first.")

    def _selected_metric(self) -> Path | None:
        files = getattr(self, "_current_metrics", [])
        name = self.metric_var.get()
        for f in files:
            if f.name == name:
                return f
        return None

    def _clear_metric_table(self) -> None:
        self.metric_tree.delete(*self.metric_tree.get_children())
        self.metric_tree["columns"] = ()

    def _show_selected_metric(self) -> None:
        path = self._selected_metric()
        self._clear_metric_table()
        self.metric_raw.delete("1.0", tk.END)
        if path is None:
            return
        raw = read_text(path)
        self.metric_raw.insert(tk.END, raw[:20000])
        if path.suffix.lower() not in (".csv", ".tsv"):
            return
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            rows = list(csv.reader(raw.splitlines(), delimiter=delim))
        except Exception:
            return
        if not rows:
            return
        header = [h.strip() for h in rows[0]]
        self.metric_tree["columns"] = header
        for col in header:
            self.metric_tree.heading(col, text=col)
            self.metric_tree.column(col, width=max(80, min(220, 8 * len(col) + 20)))
        for row in rows[1:400]:  # cap for responsiveness
            values = list(row) + [""] * (len(header) - len(row))
            self.metric_tree.insert("", tk.END, values=values[: len(header)])

    def _open_selected_metric(self) -> None:
        path = self._selected_metric()
        if path is None:
            messagebox.showinfo("No file", "Select a metric file first.")
            return
        open_path(path)

    # Logs
    def _refresh_logs(self) -> None:
        files = list_sorted(LOGS_DIR, ("*.log", "*.md", "*.txt"))
        names = [f.name for f in files]
        self.log_combo["values"] = names
        preferred = next((n for n in names if n == "benchmark_run.log"), names[0] if names else "")
        if names and self.log_file_var.get() not in names:
            self.log_file_var.set(preferred)
        self._current_logs = files
        if names:
            self._show_selected_log()
        else:
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, "No log files yet — run the benchmark first.")

    def _selected_log(self) -> Path | None:
        files = getattr(self, "_current_logs", [])
        name = self.log_file_var.get()
        for f in files:
            if f.name == name:
                return f
        return None

    def _show_selected_log(self) -> None:
        path = self._selected_log()
        self.log_text.delete("1.0", tk.END)
        if path is None:
            self.log_text.insert(tk.END, "No log selected.")
            return
        self.log_text.insert(tk.END, read_text(path))

    def _open_selected_log(self) -> None:
        path = self._selected_log()
        if path is None:
            messagebox.showinfo("No file", "Select a log file first.")
            return
        open_path(path)

    # Search
    def _fill_query(self, phrase: str) -> None:
        self.query_var.set(phrase)

    def _load_retriever(self) -> None:
        """Build the SongRetriever from the corpus (runs in a thread to keep UI responsive)."""
        if self._retriever is not None:
            self._retriever_status.set("Index already loaded.")
            return
        self._retriever_status.set("Loading corpus & building retrieval index… (a few seconds)")
        self.status_var.set("Loading retrieval index…")

        def _worker() -> None:
            try:
                import yaml
                from src.data_loader import load_raw_corpus, detect_boundaries
                from src.retrieval import SongRetriever

                if CONFIG_PATH.exists():
                    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                        config = yaml.safe_load(fh) or {}
                else:
                    config = {}
                corpus_rel = config.get("paths", {}).get("raw_corpus", "data/raw/corpus_ilocano_liturgy.txt")
                corpus_path = PROJECT_ROOT / corpus_rel
                raw = load_raw_corpus(str(corpus_path))
                units, method = detect_boundaries(raw, config)
                song_units = [u for u in units if u.unit_type in ("hymn", "canticle")]
                if not song_units:
                    song_units = units
                retriever = SongRetriever(song_units)
                self._retriever = retriever
                n = len(song_units)
                self.after(
                    0,
                    lambda: (
                        self._retriever_status.set(f"Index ready — {n} songs (boundary method: {method})."),
                        self.status_var.set("Retrieval index ready"),
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda: (
                        self._retriever_status.set(f"Failed to load index: {exc}"),
                        self.status_var.set("Retrieval index failed"),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _do_search(self) -> None:
        phrase = self.query_var.get().strip()
        if not phrase:
            messagebox.showinfo("Empty query", "Type a phrase first.")
            return
        if self._retriever is None:
            self._load_retriever()
            messagebox.showinfo("Loading", "Retrieval index is loading — click Search again in a moment.")
            return
        try:
            k = int(self.topk_var.get())
        except ValueError:
            k = 5
        results = self._retriever.search(phrase, k=k)
        self.search_tree.delete(*self.search_tree.get_children())
        for i, (song_id, title, score) in enumerate(results, 1):
            self.search_tree.insert("", tk.END, values=(i, song_id, title, f"{score:.4f}"))
        self.status_var.set(f"Search: {len(results)} results for '{phrase}'")


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    # Ensure we are running from the project root so `python -m src...` works
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    app = BenchmarkApp()
    app.mainloop()


if __name__ == "__main__":
    main()
