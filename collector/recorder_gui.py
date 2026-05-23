"""
collector/recorder_gui.py
=========================
Standalone GUI recorder for BehaviorDNA.
Compiles to a single .exe via PyInstaller — no Python required for end users.

Build command (run on Windows):
    pip install pyinstaller pynput
    pyinstaller --onefile --windowed --name BehaviorDNA_Recorder recorder_gui.py

The .exe will appear in dist/BehaviorDNA_Recorder.exe
"""

import json
import sys
import time
import tkinter as tk
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

try:
    from pynput import keyboard, mouse
except ImportError:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing dependency", "pynput not found.\nRun: pip install pynput"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Output directory — saves next to the .exe
# ---------------------------------------------------------------------------

if getattr(sys, "frozen", False):
    # Running as compiled .exe
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent.parent

OUTPUT_DIR = BASE_DIR / "sessions"


# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------

events: list[dict] = []
recording = False
start_time: float = 0.0
prev_x: int | None = None
prev_y: int | None = None
mouse_listener = None
keyboard_listener = None


def ts() -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)


# --- Mouse ---


def on_move(x, y):
    if not recording:
        return
    global prev_x, prev_y
    dx = x - prev_x if prev_x is not None else 0
    dy = y - prev_y if prev_y is not None else 0
    prev_x, prev_y = x, y
    events.append({"t": ts(), "type": "mouse_move", "x": x, "y": y, "dx": dx, "dy": dy})


def on_click(x, y, button, pressed):
    if not recording:
        return
    events.append(
        {
            "t": ts(),
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": str(button),
            "pressed": pressed,
        }
    )


def on_scroll(x, y, dx, dy):
    if not recording:
        return
    events.append(
        {"t": ts(), "type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy}
    )


# --- Keyboard ---


def _key_name(key) -> str:
    try:
        return key.char or str(key)
    except AttributeError:
        return str(key)


def on_key_press(key):
    if not recording:
        return
    events.append({"t": ts(), "type": "key_press", "key": _key_name(key)})


def on_key_release(key):
    if not recording:
        return
    events.append({"t": ts(), "type": "key_release", "key": _key_name(key)})


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_session(player, game, activity, sensitivity, dpi) -> Path:
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp}_{player}_{game}_{session_id}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    session = {
        "session_id": session_id,
        "player": player,
        "game": game,
        "activity": activity,
        "sensitivity": sensitivity,
        "dpi": dpi,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": ts(),
        "event_count": len(events),
        "events": events,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session, f, separators=(",", ":"))
    return output_path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#e94560"
ACCENT_HOVER = "#ff6b81"
TEXT = "#eaeaea"
TEXT_DIM = "#8892a4"
INPUT_BG = "#0f3460"
SUCCESS = "#4ecca3"
WARNING = "#f5a623"

GAMES = ["Valorant", "CS2", "GTA5", "Tarkov", "Arc Raiders", "Other"]
ACTIVITIES = ["on_foot", "driving", "combat", "sniping", "free_roam"]


class RecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BehaviorDNA Recorder")
        self.root.geometry("420x560")
        self.root.resizable(False, False)
        self.root.configure(bg=DARK_BG)

        self._center_window()
        self._build_ui()

        self.timer_job = None
        self.elapsed_seconds = 0

    def _center_window(self):
        self.root.update_idletasks()
        w, h = 420, 560
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # --- Header ---
        header = tk.Frame(self.root, bg=ACCENT, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_font = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        tk.Label(
            header,
            text="🧬  BehaviorDNA Recorder",
            font=title_font,
            bg=ACCENT,
            fg="white",
        ).pack(expand=True)

        # --- Form panel ---
        form = tk.Frame(self.root, bg=PANEL_BG, padx=28, pady=20)
        form.pack(fill="x", padx=16, pady=(16, 0))

        label_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        input_font = tkfont.Font(family="Segoe UI", size=11)

        def field(parent, label, row, placeholder="", width=32):
            tk.Label(
                parent, text=label, font=label_font, bg=PANEL_BG, fg=TEXT_DIM
            ).grid(row=row * 2, column=0, sticky="w", pady=(8, 1))
            var = tk.StringVar()
            entry = tk.Entry(
                parent,
                textvariable=var,
                font=input_font,
                bg=INPUT_BG,
                fg=TEXT,
                insertbackground=TEXT,
                relief="flat",
                bd=0,
                width=width,
            )
            entry.grid(row=row * 2 + 1, column=0, sticky="ew", ipady=7)
            if placeholder:
                entry.insert(0, placeholder)
                entry.config(fg=TEXT_DIM)

                def on_focus_in(e, en=entry, ph=placeholder, v=var):
                    if en.get() == ph:
                        en.delete(0, "end")
                        en.config(fg=TEXT)

                def on_focus_out(e, en=entry, ph=placeholder):
                    if not en.get():
                        en.insert(0, ph)
                        en.config(fg=TEXT_DIM)

                entry.bind("<FocusIn>", on_focus_in)
                entry.bind("<FocusOut>", on_focus_out)
            return var

        self.player_var = field(form, "YOUR NAME", 0, placeholder="e.g. jiri")

        # Game dropdown
        tk.Label(form, text="GAME", font=label_font, bg=PANEL_BG, fg=TEXT_DIM).grid(
            row=2, column=0, sticky="w", pady=(8, 1)
        )
        self.game_var = tk.StringVar(value=GAMES[0])
        game_menu = tk.OptionMenu(form, self.game_var, *GAMES)
        game_menu.config(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground="white",
            relief="flat",
            font=input_font,
            bd=0,
            highlightthickness=0,
            indicatoron=True,
            width=28,
        )
        game_menu["menu"].config(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground="white",
            font=input_font,
        )
        game_menu.grid(row=3, column=0, sticky="ew", ipady=4)

        # Activity dropdown
        tk.Label(form, text="ACTIVITY", font=label_font, bg=PANEL_BG, fg=TEXT_DIM).grid(
            row=4, column=0, sticky="w", pady=(8, 1)
        )
        self.activity_var = tk.StringVar(value=ACTIVITIES[0])
        activity_menu = tk.OptionMenu(form, self.activity_var, *ACTIVITIES)
        activity_menu.config(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground="white",
            relief="flat",
            font=input_font,
            bd=0,
            highlightthickness=0,
            indicatoron=True,
            width=28,
        )
        activity_menu["menu"].config(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground="white",
            font=input_font,
        )
        activity_menu.grid(row=5, column=0, sticky="ew", ipady=4)

        # Sens + DPI side by side
        tk.Label(
            form, text="IN-GAME SENSITIVITY", font=label_font, bg=PANEL_BG, fg=TEXT_DIM
        ).grid(row=6, column=0, sticky="w", pady=(8, 1))
        sens_dpi = tk.Frame(form, bg=PANEL_BG)
        sens_dpi.grid(row=7, column=0, sticky="ew")

        self.sens_var = tk.StringVar()
        sens_entry = tk.Entry(
            sens_dpi,
            textvariable=self.sens_var,
            font=input_font,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            width=13,
        )
        sens_entry.pack(side="left", ipady=7)
        sens_entry.insert(0, "e.g. 0.45")
        sens_entry.config(fg=TEXT_DIM)

        def sens_in(e):
            if sens_entry.get() == "e.g. 0.45":
                sens_entry.delete(0, "end")
                sens_entry.config(fg=TEXT)

        def sens_out(e):
            if not sens_entry.get():
                sens_entry.insert(0, "e.g. 0.45")
                sens_entry.config(fg=TEXT_DIM)

        sens_entry.bind("<FocusIn>", sens_in)
        sens_entry.bind("<FocusOut>", sens_out)

        tk.Label(
            sens_dpi, text="  DPI", font=label_font, bg=PANEL_BG, fg=TEXT_DIM
        ).pack(side="left", padx=(16, 4))
        self.dpi_var = tk.StringVar()
        dpi_entry = tk.Entry(
            sens_dpi,
            textvariable=self.dpi_var,
            font=input_font,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            width=10,
        )
        dpi_entry.pack(side="left", ipady=7)
        dpi_entry.insert(0, "e.g. 800")
        dpi_entry.config(fg=TEXT_DIM)

        def dpi_in(e):
            if dpi_entry.get() == "e.g. 800":
                dpi_entry.delete(0, "end")
                dpi_entry.config(fg=TEXT)

        def dpi_out(e):
            if not dpi_entry.get():
                dpi_entry.insert(0, "e.g. 800")
                dpi_entry.config(fg=TEXT_DIM)

        dpi_entry.bind("<FocusIn>", dpi_in)
        dpi_entry.bind("<FocusOut>", dpi_out)

        form.columnconfigure(0, weight=1)

        # --- Status panel ---
        status_frame = tk.Frame(self.root, bg=DARK_BG, pady=12)
        status_frame.pack(fill="x", padx=16)

        self.status_var = tk.StringVar(value="Ready to record")
        status_font = tkfont.Font(family="Segoe UI", size=10)
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=status_font,
            bg=DARK_BG,
            fg=TEXT_DIM,
        )
        self.status_label.pack()

        self.timer_var = tk.StringVar(value="")
        timer_font = tkfont.Font(family="Segoe UI Semibold", size=22, weight="bold")
        self.timer_label = tk.Label(
            status_frame,
            textvariable=self.timer_var,
            font=timer_font,
            bg=DARK_BG,
            fg=TEXT,
        )
        self.timer_label.pack()

        self.events_var = tk.StringVar(value="")
        events_font = tkfont.Font(family="Segoe UI", size=9)
        tk.Label(
            status_frame,
            textvariable=self.events_var,
            font=events_font,
            bg=DARK_BG,
            fg=TEXT_DIM,
        ).pack()

        # --- Button ---
        btn_frame = tk.Frame(self.root, bg=DARK_BG)
        btn_frame.pack(fill="x", padx=16, pady=(4, 16))

        btn_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.btn = tk.Button(
            btn_frame,
            text="▶  START RECORDING",
            font=btn_font,
            bg=ACCENT,
            fg="white",
            activebackground=ACCENT_HOVER,
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._toggle,
        )
        self.btn.pack(fill="x", ipady=14)

        # --- Footer ---
        footer_font = tkfont.Font(family="Segoe UI", size=8)
        tk.Label(
            self.root,
            text=f"Sessions saved to:  {OUTPUT_DIR}",
            font=footer_font,
            bg=DARK_BG,
            fg=TEXT_DIM,
            wraplength=390,
        ).pack(side="bottom", pady=8)

    # -----------------------------------------------------------------------
    # Logic
    # -----------------------------------------------------------------------

    def _validate(self) -> bool:
        player = self.player_var.get().strip()
        sens_raw = self.sens_var.get().strip()
        dpi_raw = self.dpi_var.get().strip()

        if not player or player == "e.g. jiri":
            messagebox.showwarning("Missing field", "Please enter your name.")
            return False
        try:
            float(sens_raw)
        except ValueError:
            messagebox.showwarning(
                "Invalid sensitivity", "Sensitivity must be a number (e.g. 0.45)."
            )
            return False
        try:
            int(dpi_raw)
        except ValueError:
            messagebox.showwarning(
                "Invalid DPI", "DPI must be a whole number (e.g. 800)."
            )
            return False
        return True

    def _toggle(self):
        if not recording:
            self._start()
        else:
            self._stop()

    def _start(self):
        global recording, start_time, events, prev_x, prev_y
        global mouse_listener, keyboard_listener

        if not self._validate():
            return

        events = []
        prev_x, prev_y = None, None
        start_time = time.perf_counter()
        recording = True
        self.elapsed_seconds = 0

        mouse_listener = mouse.Listener(
            on_move=on_move, on_click=on_click, on_scroll=on_scroll
        )
        keyboard_listener = keyboard.Listener(
            on_press=on_key_press, on_release=on_key_release
        )
        mouse_listener.start()
        keyboard_listener.start()

        self.btn.config(text="⏹  STOP RECORDING", bg="#2d2d44")
        self.status_label.config(fg=SUCCESS)
        self.status_var.set("🔴  Recording — switch to your game!")
        self._tick()

    def _tick(self):
        if not recording:
            return
        self.elapsed_seconds += 1
        m, s = divmod(self.elapsed_seconds, 60)
        self.timer_var.set(f"{m:02d}:{s:02d}")
        self.events_var.set(f"{len(events):,} events captured")
        self.timer_job = self.root.after(1000, self._tick)

    def _stop(self):
        global recording, mouse_listener, keyboard_listener

        recording = False
        if mouse_listener:
            mouse_listener.stop()
        if keyboard_listener:
            keyboard_listener.stop()
        if self.timer_job:
            self.root.after_cancel(self.timer_job)

        if len(events) < 100:
            messagebox.showwarning(
                "Few events captured",
                "Very few events were recorded.\n"
                "Make sure the game window had focus during recording.",
            )

        player = self.player_var.get().strip()
        game = self.game_var.get().strip().lower().replace(" ", "_")
        activity = self.activity_var.get().strip()
        sens = float(self.sens_var.get().strip())
        dpi = int(self.dpi_var.get().strip())

        output_path = save_session(player, game, activity, sens, dpi)

        self.btn.config(text="▶  START RECORDING", bg=ACCENT)
        self.status_label.config(fg=SUCCESS)
        self.status_var.set(f"✅  Saved: {output_path.name}")
        self.events_var.set(f"{len(events):,} events  •  {self.elapsed_seconds}s")
        self.timer_var.set("")

        messagebox.showinfo(
            "Session saved!",
            f"Session recorded successfully.\n\n"
            f"Player   : {player}\n"
            f"Game     : {game}\n"
            f"Activity : {activity}\n"
            f"Events   : {len(events):,}\n\n"
            f"File saved to:\n{output_path}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    root = tk.Tk()
    app = RecorderApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
