import tkinter as tk
from tkinter import messagebox
import winsound

# ================= 科学时间配置 (分钟) =================
# 哥伦比亚大学运动医学黄金配比：25分钟专注坐姿 + 5分钟站立微休息
SIT_MINUTES = 25
STAND_MINUTES = 5
# =======================================================


class SitStandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 坐站交替助手")
        self.root.geometry("330x250")
        self.root.resizable(False, False)

        # 默认开启窗口置顶，方便随时瞥一眼时间
        self.is_topmost = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)

        # 核心状态变量
        self.is_sitting = True  # True: 坐姿阶段, False: 站姿阶段
        self.is_running = False  # 是否正在倒计时
        self.waiting_confirm = False  # 是否在等待用户点击确认切换
        self.cycle_count = 1  # 当前循环轮次
        self.remaining_seconds = int(SIT_MINUTES * 60)
        self.timer_job = None

        # 主题配色方案
        self.colors = {
            "sit_bg": "#EEF5FF",  # 坐姿背景：护眼浅蓝
            "sit_fg": "#1E3A8A",  # 坐姿文字：深蓝
            "stand_bg": "#ECFDF5",  # 站姿背景：清新浅绿
            "stand_fg": "#065F46",  # 站姿文字：深绿
            "alert_bg": "#FEF2F2",  # 提醒背景：醒目浅红
            "alert_fg": "#991B1B",  # 提醒文字：深红
        }

        self._build_ui()
        self._apply_theme("sit")

    def _build_ui(self):
        """构建图形界面组件"""
        # 1. 顶部栏：轮次显示 + 置顶复选框
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=15, pady=(10, 0))

        self.cycle_label = tk.Label(
            self.top_frame,
            text=f"第 {self.cycle_count} 轮循环",
            font=("Microsoft YaHei", 9, "bold"),
        )
        self.cycle_label.pack(side=tk.LEFT)

        self.topmost_cb = tk.Checkbutton(
            self.top_frame,
            text="📌 窗口置顶",
            variable=self.is_topmost,
            command=self.toggle_topmost,
            font=("Microsoft YaHei", 9),
            cursor="hand2",
        )
        self.topmost_cb.pack(side=tk.RIGHT)

        # 2. 状态标题
        self.status_label = tk.Label(
            self.root, text="💻 坐姿专注阶段", font=("Microsoft YaHei", 13, "bold")
        )
        self.status_label.pack(pady=(8, 2))

        # 3. 超大号倒计时数字
        self.time_label = tk.Label(
            self.root,
            text=self._format_time(self.remaining_seconds),
            font=("Consolas", 42, "bold"),
        )
        self.time_label.pack(pady=2)

        # 4. 底部提示语
        self.hint_label = tk.Label(
            self.root,
            text=f"科学配比：坐 {SIT_MINUTES} 分钟 / 站 {STAND_MINUTES} 分钟",
            font=("Microsoft YaHei", 9),
        )
        self.hint_label.pack(pady=(0, 12))

        # 5. 按钮控制区
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=5)

        self.start_btn = tk.Button(
            self.btn_frame,
            text="▶ 开始计时",
            width=11,
            height=1,
            font=("Microsoft YaHei", 10, "bold"),
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.toggle_start_pause,
        )
        self.start_btn.grid(row=0, column=0, padx=6)

        self.skip_btn = tk.Button(
            self.btn_frame,
            text="⏭ 跳过本阶段",
            width=11,
            height=1,
            font=("Microsoft YaHei", 10),
            bg="#E2E8F0",
            fg="#334155",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.skip_stage,
        )
        self.skip_btn.grid(row=0, column=1, padx=6)

        self.reset_btn = tk.Button(
            self.btn_frame,
            text="↺ 重置",
            width=6,
            height=1,
            font=("Microsoft YaHei", 10),
            bg="#E2E8F0",
            fg="#334155",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.reset_stage,
        )
        self.reset_btn.grid(row=0, column=2, padx=6)

    def _apply_theme(self, mode):
        """切换界面配色主题 (sit / stand / alert)"""
        bg = self.colors[f"{mode}_bg"]
        fg = self.colors[f"{mode}_fg"]

        self.root.configure(bg=bg)
        self.top_frame.configure(bg=bg)
        self.btn_frame.configure(bg=bg)
        self.cycle_label.configure(bg=bg, fg=fg)
        self.topmost_cb.configure(bg=bg, fg=fg, activebackground=bg, selectcolor=bg)
        self.status_label.configure(bg=bg, fg=fg)
        self.time_label.configure(bg=bg, fg=fg)
        self.hint_label.configure(bg=bg, fg=fg)

    def _format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

    def toggle_topmost(self):
        """切换窗口是否保持在最前"""
        self.root.attributes("-topmost", self.is_topmost.get())

    def play_sound(self, is_stand_up=True):
        """播放提示音"""
        try:
            if is_stand_up:
                winsound.Beep(880, 250)
                winsound.Beep(1046, 350)
            else:
                winsound.Beep(1046, 250)
                winsound.Beep(880, 350)
        except Exception:
            self.root.bell()

    def toggle_start_pause(self):
        """开始 / 暂停 / 到点确认 按钮逻辑"""
        # 如果当前处于“到点等待确认”状态，点击按钮即开启下一阶段
        if self.waiting_confirm:
            self.waiting_confirm = False
            self._switch_stage()
            self.is_running = True
            self.start_btn.config(text="⏸ 暂停", bg="#D97706")
            self._tick()
            return

        if self.is_running:
            # 暂停计时
            self.is_running = False
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
            self.start_btn.config(text="▶ 继续", bg="#2563EB")
            self.hint_label.config(text="⏸ 计时已暂停")
        else:
            # 开始计时
            self.is_running = True
            self.start_btn.config(text="⏸ 暂停", bg="#D97706")
            self.hint_label.config(
                text="专注工作学习中..."
                if self.is_sitting
                else "请站立或走动活动一下肩颈腰背~"
            )
            self._tick()

    def _tick(self):
        """非阻塞式每秒倒计时（使用 Tkinter 的 after 机制，界面不卡顿）"""
        if not self.is_running:
            return

        if self.remaining_seconds > 0:
            self.remaining_seconds -= 1
            self.time_label.config(text=self._format_time(self.remaining_seconds))
            self.timer_job = self.root.after(1000, self._tick)
        else:
            self._on_stage_finish()

    def _on_stage_finish(self):
        """当前阶段倒计时归零时的提醒逻辑"""
        self.is_running = False
        self.waiting_confirm = True

        # 强制唤醒窗口并弹出到最顶层
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self._apply_theme("alert")

        if self.is_sitting:
            self.play_sound(is_stand_up=True)
            self.status_label.config(text="⏰ 久坐提醒：该站起来啦！")
            self.hint_label.config(
                text=f"已坐满 {SIT_MINUTES} 分钟！请站起身点击下方按钮"
            )
            self.start_btn.config(text="🧍‍♀️ 我已站起", bg="#DC2626")
        else:
            self.play_sound(is_stand_up=False)
            self.status_label.config(text="🪑 站立达标：可以坐回啦！")
            self.hint_label.config(
                text=f"已站满 {STAND_MINUTES} 分钟！请坐回椅子点击下方按钮"
            )
            self.start_btn.config(text="💻 我已坐好", bg="#059669")

        # 恢复用户原本的置顶设置偏好
        self.root.attributes("-topmost", self.is_topmost.get())

    def _switch_stage(self):
        """在坐姿和站姿之间切换"""
        if self.is_sitting:
            # 切换到站姿
            self.is_sitting = False
            self.remaining_seconds = int(STAND_MINUTES * 60)
            self._apply_theme("stand")
            self.status_label.config(text="🌱 站立/活动阶段")
            self.hint_label.config(text="请站立或走动，缓解下肢与腰椎压力~")
        else:
            # 切换到坐姿，轮次+1
            self.is_sitting = True
            self.cycle_count += 1
            self.cycle_label.config(text=f"第 {self.cycle_count} 轮循环")
            self.remaining_seconds = int(SIT_MINUTES * 60)
            self._apply_theme("sit")
            self.status_label.config(text="💻 坐姿专注阶段")
            self.hint_label.config(text="专注工作学习中...")

        self.time_label.config(text=self._format_time(self.remaining_seconds))

    def skip_stage(self):
        """手动跳过当前阶段，直接进入下一阶段待命"""
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.is_running = False
        self.waiting_confirm = False
        self._switch_stage()
        self.start_btn.config(text="▶ 开始计时", bg="#2563EB")

    def reset_stage(self):
        """重置当前阶段的倒计时"""
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.is_running = False
        self.waiting_confirm = False
        self.remaining_seconds = int(
            (SIT_MINUTES if self.is_sitting else STAND_MINUTES) * 60
        )
        self.time_label.config(text=self._format_time(self.remaining_seconds))
        self._apply_theme("sit" if self.is_sitting else "stand")
        self.status_label.config(
            text="💻 坐姿专注阶段" if self.is_sitting else "🌱 站立/活动阶段"
        )
        self.hint_label.config(text="计时已重置，点击开始启动")
        self.start_btn.config(text="▶ 开始计时", bg="#2563EB")


if __name__ == "__main__":
    root = tk.Tk()
    app = SitStandApp(root)
    root.mainloop()
