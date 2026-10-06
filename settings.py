from os.path import isfile
from json import dump, load

CONFIG_PATH = "config.json"

if not isfile(CONFIG_PATH):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        dump({}, f)
        data = {}
else:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data: dict[str, any] = load(f)

_cget = data.get

class SETTINGS:
    n_mahjong = _cget("n_mahjong", 40)
    enable_msaa = _cget("enable_msaa", False)
    enable_vsync = _cget("enable_vsyc", True)
    target_fps = _cget("target_fps", 60)
    use_mipmap = _cget("use_mipmap", True)
    use_anisotropy = _cget("use_anisotropy", False)
    round_n_hint = _cget("round_n_hint", 32)
    shadow_mode = _cget("shadow_mode", 1)
    pcf_samples = _cget("pcf_samples", 16)
    physics_sub = _cget("physics_sub", 2)

if __name__ == "__main__":
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import BOTH, X, LEFT, RIGHT, YES, W

    DEFAULTS = {
        "n_mahjong": 40,
        "enable_msaa": False,
        "enable_vsync": True,
        "target_fps": 60,
        "use_mipmap": True,
        "use_anisotropy": False,
        "round_n_hint": 32,
        "shadow_mode": 1,
        "pcf_samples": 16,
        "physics_sub": 2
    }

    N_MAHJONG_LEVELS = (("8 个", 8), ("40 个", 40), ("200 个", 200), ("1000 个", 1000), ("5000 个", 5000))
    ROUND_LEVELS = (("低", 16), ("流畅", 32), ("中等", 64), ("高品质", 128), ("极佳 !", 256))
    PCF_LEVELS = (("低", 8), ("流畅", 16), ("中等", 32), ("高品质", 64), ("极佳 !", 128))
    SHADOW_MODES = (("无阴影", 0), ("硬边阴影", 1), ("柔和阴影", 2))
    PHYSICS_SUB = (("低", 1), ("流畅", 2), ("中等", 4), ("高品质", 8), ("极佳 !", 12))

    N_MAHJONG_VALUES = [v for _, v in N_MAHJONG_LEVELS]
    ROUND_VALUES = [v for _, v in ROUND_LEVELS]
    PHYSICS_VALUES = [v for _, v in PHYSICS_SUB]
    PCF_VALUES = [v for _, v in PCF_LEVELS]

    def load_config() -> dict:
        raw = {}
        if isfile(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    data = load(f)
                if isinstance(data, dict):
                    raw = data
            except (OSError, ValueError):
                raw = {}

        cfg = dict(DEFAULTS)
        for key in DEFAULTS:
            if key in raw:
                cfg[key] = raw[key]
        return cfg

    def save_config(cfg: dict) -> None:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            dump({k: cfg[k] for k in DEFAULTS}, f, indent=4, ensure_ascii=False)

    def nearest(value, choices):
        """把配置里的任意数值吸附到最近的合法档位"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return choices[0]
        return min(choices, key=lambda c: abs(c - value))

    def clamp_fps(value) -> int:
        try:
            v = int(round(float(value) / 10.0) * 10)
        except (TypeError, ValueError):
            v = 60
        return max(30, min(180, v))

    class SettingsWindow(ttk.Window):
        def __init__(self):
            super().__init__()
            self.title("图形设置")

            cfg = load_config()

            # ---- 变量 -------------------------------------------------------- #
            self.n_mahjong_var = ttk.IntVar(value=nearest(cfg["n_mahjong"], N_MAHJONG_VALUES))
            self.msaa_var = ttk.BooleanVar(value=bool(cfg["enable_msaa"]))
            self.vsync_var = ttk.BooleanVar(value=bool(cfg["enable_vsync"]))
            self.mipmap_var = ttk.BooleanVar(value=bool(cfg["use_mipmap"]))
            self.aniso_var = ttk.BooleanVar(value=bool(cfg["use_anisotropy"]))

            self.fps_var = ttk.DoubleVar(value=float(clamp_fps(cfg["target_fps"])))

            self.round_var = ttk.IntVar(value=nearest(cfg["round_n_hint"], ROUND_VALUES))
            self.physics_var = ttk.IntVar(value=nearest(cfg["physics_sub"], PHYSICS_VALUES))

            try:
                shadow = int(cfg["shadow_mode"])
            except (TypeError, ValueError):
                shadow = 1
            self.shadow_var = ttk.IntVar(value=shadow if shadow in (0, 1, 2) else 1)

            self.pcf_var = ttk.IntVar(value=nearest(cfg["pcf_samples"], PCF_VALUES))

            # ---- 布局 -------------------------------------------------------- #
            root = ttk.Frame(self, padding=16)
            root.pack(fill=BOTH, expand=YES)

            self._build_fps(root)
            self._build_entities(root)
            self._build_round(root)
            self._build_shadow(root)
            self._build_physics(root)
            self._build_footer(root)

            self._sync_pcf_state()
            self._update_fps_label()

            self.withdraw()
            self.update()
            w = self.winfo_width() * 1.5
            h = self.winfo_height()
            self.geometry(f"{int(w)}x{h}")
            self.resizable(False, False)
            self.deiconify()
            self.place_window_center()

        # ------------------------------------------------------------------ #
        def _build_fps(self, parent):
            box = ttk.Labelframe(parent, text="  显示  ", padding=(12, 10))
            box.pack(fill=X, pady=(0, 10))

            head = ttk.Frame(box)
            head.pack(fill=X)
            ttk.Label(head, text="最大帧率").pack(side=LEFT)
            self.fps_label = ttk.Label(head, text="", bootstyle="info")
            self.fps_label.pack(side=RIGHT)

            ttk.Scale(
                box,
                from_=30,
                to=180,
                orient="horizontal",
                variable=self.fps_var,
                command=self._on_fps,
                bootstyle="info",
            ).pack(fill=X, pady=(4, 0))

            ttk.Checkbutton(
                box, text="垂直同步 (VSync)", variable=self.vsync_var, bootstyle="round-toggle"
            ).pack(anchor="w", pady=(6, 0))
            ttk.Checkbutton(
                box, text="多重采样抗锯齿 (MSAA)", variable=self.msaa_var, bootstyle="round-toggle"
            ).pack(anchor="w", pady=(6, 0))

        def _build_entities(self, parent):
            box = ttk.Labelframe(parent, text="  实体与纹理  ", padding=(12, 10))
            box.pack(fill=X, pady=(0, 10))
            frm = ttk.Frame(box)
            frm.pack(fill=X, pady=(0, 10))
            self._row_of_radios(frm, N_MAHJONG_LEVELS, self.n_mahjong_var, "mahjong")
            ttk.Checkbutton(
                box,
                text="多级渐远纹理",
                variable=self.mipmap_var,
                bootstyle="round-toggle"
            ).pack(anchor=W)
            ttk.Checkbutton(
                box,
                text="各向异性过滤",
                variable=self.aniso_var,
                bootstyle="round-toggle",
            ).pack(anchor=W, pady=(6, 0))

        @staticmethod
        def _row_of_radios(row: ttk.Frame, levels, variable, uniform, command=None):
            for col in range(len(levels)):
                row.columnconfigure(col, weight=1, uniform=uniform)

            buttons = []
            for col, (text, value) in enumerate(levels):
                rb = ttk.Radiobutton(
                    row,
                    text=text,
                    value=value,
                    variable=variable,
                    bootstyle="toolbutton",
                    command=command,
                )
                rb.grid(row=0, column=col, sticky="ew", padx=3)
                buttons.append(rb)
            return buttons

        def _build_round(self, parent):
            box = ttk.Labelframe(parent, text="  圆角级别  ", padding=(12, 10))
            box.pack(fill=X, pady=(0, 10))

            row = ttk.Frame(box)
            row.pack(fill=X)
            self._row_of_radios(row, ROUND_LEVELS, self.round_var, "round")

        def _build_shadow(self, parent):
            box = ttk.Labelframe(parent, text="  阴影  ", padding=(12, 10))
            box.pack(fill=X, pady=(0, 10))

            row = ttk.Frame(box)
            row.pack(fill=X)
            self._row_of_radios(
                row, SHADOW_MODES, self.shadow_var, "shadow", command=self._sync_pcf_state
            )

            self.pcf_box = ttk.Labelframe(parent, text="  PCF 采样  ", padding=(12, 10))
            self.pcf_box.pack(fill=X, pady=(0, 10))

            prow = ttk.Frame(self.pcf_box)
            prow.pack(fill=X)
            self.pcf_buttons = self._row_of_radios(prow, PCF_LEVELS, self.pcf_var, "pcf")

            self.pcf_hint = ttk.Label(self.pcf_box, text="", bootstyle="secondary")
            self.pcf_hint.pack(anchor=W, pady=(6, 0))

        def _build_physics(self, parent):
            box = ttk.Labelframe(parent, text="  物理引擎性能  ", padding=(12, 10))
            box.pack(fill=X, pady=(0, 10))

            row = ttk.Frame(box)
            row.pack(fill=X)
            self._row_of_radios(row, PHYSICS_SUB, self.physics_var, "physics")

        def _build_footer(self, parent):
            row = ttk.Frame(parent)
            row.pack(fill=X, pady=(4, 0))

            self.status = ttk.Label(row, text=f"配置文件：{CONFIG_PATH}", bootstyle="secondary")
            self.status.pack(side=LEFT)

            ttk.Button(
                row, text="恢复默认", bootstyle="secondary-outline", command=self._reset
            ).pack(side=RIGHT)
            ttk.Button(
                row, text="保存", bootstyle="success", command=self._save
            ).pack(side=RIGHT, padx=(0, 8))

        def _on_fps(self, value):
            """滑块值吸附到 10 的整数倍"""
            v = int(round(float(value) / 10.0) * 10)
            v = max(30, min(180, v))
            if abs(float(value) - v) > 1e-6:
                self.fps_var.set(float(v))  # 再触发一次，但此时已对齐，不会递归
            self.fps_label.configure(text=f"{v} FPS")

        def _update_fps_label(self):
            self.fps_label.configure(text=f"{clamp_fps(self.fps_var.get())} FPS")

        def _sync_pcf_state(self):
            """仅“柔和阴影”时可调 pcf_samples"""
            soft = self.shadow_var.get() == 2
            state = "normal" if soft else "disabled"
            for rb in self.pcf_buttons:
                rb.configure(state=state)
            self.pcf_hint.configure(
                text="☑ 现在可以调整" if soft else "☒ 仅“柔和阴影”下可调整",
                bootstyle="success" if soft else "danger"
            )

        def _collect(self) -> dict:
            return {
                "n_mahjong": int(self.n_mahjong_var.get()),
                "enable_msaa": bool(self.msaa_var.get()),
                "enable_vsync": bool(self.vsync_var.get()),
                "target_fps": clamp_fps(self.fps_var.get()),
                "use_mipmap": bool(self.mipmap_var.get()),
                "use_anisotropy": bool(self.aniso_var.get()),
                "round_n_hint": int(self.round_var.get()),
                "shadow_mode": int(self.shadow_var.get()),
                "pcf_samples": int(self.pcf_var.get()),
                "physics_sub": int(self.physics_var.get()),
            }

        def _save(self):
            cfg = self._collect()
            try:
                save_config(cfg)
            except OSError as exc:
                self.status.configure(text=f"保存失败：{exc}", bootstyle="danger")
                return
            self.status.configure(
                text=f"已保存到 {CONFIG_PATH}", bootstyle="success"
            )

        def _reset(self):
            self.msaa_var.set(DEFAULTS["enable_msaa"])
            self.vsync_var.set(DEFAULTS["enable_vsync"])
            self.mipmap_var.set(DEFAULTS["use_mipmap"])
            self.aniso_var.set(DEFAULTS["use_anisotropy"])
            self.fps_var.set(float(DEFAULTS["target_fps"]))
            self.round_var.set(DEFAULTS["round_n_hint"])
            self.shadow_var.set(DEFAULTS["shadow_mode"])
            self.pcf_var.set(DEFAULTS["pcf_samples"])
            self._update_fps_label()
            self._sync_pcf_state()
            self.status.configure(text="已恢复默认值（尚未保存）", bootstyle="warning")
    
    SettingsWindow().mainloop()
