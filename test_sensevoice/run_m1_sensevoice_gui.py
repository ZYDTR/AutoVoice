import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import queue
import re
import traceback

# ================= 配置区域 =================
# 支持的模型
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"  # SenseVoice 模型（不支持 speaker diarization）
PARAFORMER_MODEL = "paraformer-zh"  # Paraformer 模型（支持 speaker diarization）

DEVICE = "cpu"
THREADS = 4
DEFAULT_OUTPUT_DIR = "/Users/zhengyidi/AutoVoice/recordings"  # 默认输出目录
SPK_MODEL = "cam++"  # 说话人识别模型
# ===========================================

class AudioTranscriptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("音频转录工具 - FunASR")
        self.root.geometry("900x700")
        
        # 变量
        self.selected_files = []
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.use_default_dir = tk.BooleanVar(value=True)
        self.processing_mode = tk.StringVar(value="direct")  # "direct" 或 "cascaded"
        self.model_type = tk.StringVar(value="sensevoice")  # "sensevoice" 或 "paraformer"（仅在 direct 模式下有效）
        self.enable_speaker = tk.BooleanVar(value=False)  # 根据模型类型动态启用/禁用（仅在 direct 模式下有效）
        self.model = None
        self.paraformer_model = None  # 级联模式需要
        self.sensevoice_model = None  # 级联模式需要
        self.is_processing = False
        self.is_paused = False
        self.should_stop = False
        self.processing_file_index = 0
        self.processing_start_time = None
        self.heartbeat_thread = None
        self.heartbeat_stop = threading.Event()
        self.processing_thread = None
        
        # 创建界面
        self.create_widgets()
        
        # 在后台加载模型（延迟一下，确保界面已创建）
        self.root.after(100, self.load_model_async)
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🎤 音频转录工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="选择音频文件", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="选择文件", command=self.select_files).grid(row=0, column=0, padx=5)
        self.file_listbox = tk.Listbox(file_frame, height=4, selectmode=tk.EXTENDED)
        self.file_listbox.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        scrollbar_files = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar_files.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.file_listbox.config(yscrollcommand=scrollbar_files.set)
        
        ttk.Button(file_frame, text="清空", command=self.clear_files).grid(row=0, column=3, padx=5)
        
        # 输出目录选择区域
        output_frame = ttk.LabelFrame(main_frame, text="输出目录", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(1, weight=1)
        
        self.use_default_check = ttk.Checkbutton(
            output_frame, 
            text="使用默认目录", 
            variable=self.use_default_dir,
            command=self.toggle_output_dir
        )
        self.use_default_check.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.output_dir_entry = ttk.Entry(output_frame, state="readonly")
        self.output_dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.output_dir_entry.config(state="normal")
        self.output_dir_entry.insert(0, DEFAULT_OUTPUT_DIR)
        self.output_dir_entry.config(state="readonly")
        
        ttk.Button(output_frame, text="浏览", command=self.select_output_dir).grid(row=0, column=2, padx=5)
        
        # 处理模式选择区域
        mode_frame = ttk.LabelFrame(main_frame, text="处理模式", padding="10")
        mode_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="直接模式（单一模型）",
            variable=self.processing_mode,
            value="direct",
            command=self.on_processing_mode_changed
        ).grid(row=0, column=0, padx=10, sticky=tk.W)
        
        ttk.Radiobutton(
            mode_frame,
            text="级联模式（Paraformer + SenseVoice）",
            variable=self.processing_mode,
            value="cascaded",
            command=self.on_processing_mode_changed
        ).grid(row=0, column=1, padx=10, sticky=tk.W)
        
        self.mode_info_label = ttk.Label(
            mode_frame,
            text="级联模式：先用 Paraformer 做说话人区分，再用 SenseVoice 识别文本（保留情感标签）",
            foreground="gray",
            font=("Arial", 9)
        )
        self.mode_info_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # 模型选择区域（仅在直接模式下显示）
        self.model_frame = ttk.LabelFrame(main_frame, text="模型选择（仅直接模式）", padding="10")
        self.model_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            self.model_frame,
            text="SenseVoice (不支持说话人区分)",
            variable=self.model_type,
            value="sensevoice",
            command=self.on_model_type_changed
        ).grid(row=0, column=0, padx=10, sticky=tk.W)
        
        ttk.Radiobutton(
            self.model_frame,
            text="Paraformer (支持说话人区分)",
            variable=self.model_type,
            value="paraformer",
            command=self.on_model_type_changed
        ).grid(row=0, column=1, padx=10, sticky=tk.W)
        
        # 说话人区分选项（仅在直接模式下显示）
        self.speaker_frame = ttk.LabelFrame(main_frame, text="说话人区分（仅直接模式）", padding="10")
        self.speaker_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.speaker_check = ttk.Checkbutton(
            self.speaker_frame,
            text="启用说话人区分 (Speaker Diarization)",
            variable=self.enable_speaker,
            command=self.on_speaker_changed
        )
        self.speaker_check.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.speaker_info_label = ttk.Label(
            self.speaker_frame,
            text="选择 Paraformer 模型后可启用",
            foreground="gray",
            font=("Arial", 9)
        )
        self.speaker_info_label.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        # 按钮组
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT)
        
        def start_processing_wrapper():
            """包装函数，添加调试日志"""
            print("\n[DEBUG] ========================================")
            print("[DEBUG] 按钮被点击！")
            print(f"[DEBUG] 按钮状态: {self.process_btn.cget('state')}")
            print(f"[DEBUG] 按钮文本: {self.process_btn.cget('text')}")
            print(f"[DEBUG] 当前处理模式: {self.processing_mode.get()}")
            print(f"[DEBUG] 已选择文件数: {len(self.selected_files) if self.selected_files else 0}")
            print(f"[DEBUG] 是否正在处理: {self.is_processing}")
            print("[DEBUG] ========================================\n")
            
            # 如果按钮是 disabled，直接返回并提示
            if self.process_btn.cget('state') == 'disabled':
                print("[DEBUG] ⚠️ 按钮处于 disabled 状态，点击无效")
                print("[DEBUG] 检查模型状态...")
                if self.processing_mode.get() == "cascaded":
                    print(f"[DEBUG]   paraformer_model: {self.paraformer_model is not None}")
                    print(f"[DEBUG]   sensevoice_model: {self.sensevoice_model is not None}")
                else:
                    print(f"[DEBUG]   model: {self.model is not None}")
                messagebox.showwarning("提示", "按钮当前不可用，请检查：\n1. 模型是否已加载完成\n2. 是否已选择音频文件\n3. 是否正在处理中")
                return
            
            try:
                self.start_processing()
            except Exception as e:
                print(f"[DEBUG] start_processing 发生异常: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("错误", f"处理失败: {str(e)}\n\n详细错误请查看终端")
        
        self.process_btn = ttk.Button(
            button_frame, 
            text="开始处理", 
            command=start_processing_wrapper,
            state="disabled"
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="停止",
            command=self.stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(
            button_frame,
            text="暂停",
            command=self.pause_processing,
            state="disabled"
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.resume_btn = ttk.Button(
            button_frame,
            text="继续",
            command=self.resume_processing,
            state="disabled"
        )
        self.resume_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="正在加载模型...", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var, 
            maximum=100,
            length=400
        )
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 日志显示区域 - 使用 Notebook (标签页)
        log_notebook_frame = ttk.Frame(main_frame)
        log_notebook_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_notebook_frame.columnconfigure(0, weight=1)
        log_notebook_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # 创建 Notebook (标签页容器)
        self.log_notebook = ttk.Notebook(log_notebook_frame)
        self.log_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标签页1: 处理日志（主要信息）
        log_frame = ttk.Frame(self.log_notebook, padding="10")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_notebook.add(log_frame, text="处理日志")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.config(state="disabled")
        
        # 配置文本标签样式（子信息使用灰色）
        self.log_text.tag_config("main", foreground="black")
        self.log_text.tag_config("sub", foreground="gray60")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
        self.log_text.tag_config("info", foreground="blue")
        
        # 标签页2: 详细日志（完整错误堆栈和调试信息）
        detail_log_frame = ttk.Frame(self.log_notebook, padding="10")
        detail_log_frame.columnconfigure(0, weight=1)
        detail_log_frame.rowconfigure(0, weight=1)
        self.log_notebook.add(detail_log_frame, text="详细日志")
        
        self.detail_log_text = scrolledtext.ScrolledText(detail_log_frame, height=15, wrap=tk.WORD, font=("Courier", 9))
        self.detail_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.detail_log_text.config(state="disabled")
        
        # 配置详细日志文本标签样式
        self.detail_log_text.tag_config("error", foreground="red", font=("Courier", 9, "bold"))
        self.detail_log_text.tag_config("warning", foreground="orange")
        self.detail_log_text.tag_config("info", foreground="blue")
        self.detail_log_text.tag_config("debug", foreground="gray")
        
        # 初始化处理模式状态（在所有 UI 组件创建完成后）
        self.on_processing_mode_changed()
    
    def log(self, message, level="main"):
        """
        添加日志到处理日志标签页
        level: "main" 主信息, "sub" 子信息（缩进显示）, "error" 错误信息, "warning" 警告, "info" 信息
        """
        self.log_text.config(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        
        # 根据级别添加不同的前缀和缩进
        if level == "sub":
            prefix = "  └─ "  # 子信息使用缩进和符号
            # 使用灰色显示子信息，更轻量
            self.log_text.insert(tk.END, f"[{timestamp}] {prefix}{message}\n", "sub")
        elif level == "error":
            prefix = "❌ "  # 错误信息
            self.log_text.insert(tk.END, f"[{timestamp}] {prefix}{message}\n", "error")
        elif level == "warning":
            prefix = "⚠️ "  # 警告信息
            self.log_text.insert(tk.END, f"[{timestamp}] {prefix}{message}\n", "warning")
        elif level == "info":
            prefix = "ℹ️ "  # 信息
            self.log_text.insert(tk.END, f"[{timestamp}] {prefix}{message}\n", "info")
        else:
            prefix = ""  # 主信息正常显示
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", "main")
        
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()
    
    def log_detail(self, message, level="info"):
        """
        添加详细日志到详细日志标签页
        level: "error", "warning", "info", "debug"
        """
        self.detail_log_text.config(state="normal")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 根据级别添加不同的前缀
        if level == "error":
            prefix = "[ERROR] "
            tag = "error"
        elif level == "warning":
            prefix = "[WARNING] "
            tag = "warning"
        elif level == "debug":
            prefix = "[DEBUG] "
            tag = "debug"
        else:
            prefix = "[INFO] "
            tag = "info"
        
        self.detail_log_text.insert(tk.END, f"[{timestamp}] {prefix}{message}\n", tag)
        self.detail_log_text.see(tk.END)
        self.detail_log_text.config(state="disabled")
        self.root.update_idletasks()
        
        # 如果是错误，自动切换到详细日志标签页
        if level == "error":
            self.log_notebook.select(1)  # 切换到详细日志标签页
    
    def select_files(self):
        """选择音频文件"""
        # 设置初始目录为默认输出目录（通常也是音频文件所在目录）
        initialdir = self.output_dir if os.path.exists(self.output_dir) else os.path.expanduser("~")
        
        # macOS 文件选择对话框的标签过滤和文件类型过滤可能冲突
        # 解决方案：将"所有文件"放在第一位，或者使用更灵活的文件类型过滤
        # 注意：macOS 的标签过滤功能是系统级别的，当用户选择标签时，
        # 如果文件类型过滤太严格，可能会导致没有文件显示
        files = filedialog.askopenfilenames(
            title="选择音频文件",
            initialdir=initialdir,
            filetypes=[
                ("所有文件", "*.*"),  # 将"所有文件"放在第一位，方便标签过滤
                ("音频文件", "*.webm *.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("WebM 文件", "*.webm"),
                ("MP3 文件", "*.mp3"),
                ("WAV 文件", "*.wav"),
                ("M4A 文件", "*.m4a"),
                ("FLAC 文件", "*.flac"),
            ]
        )
        if files:
            self.selected_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for file in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(file))
            self.log(f"已选择 {len(self.selected_files)} 个文件")
            self.update_process_button()
    
    def clear_files(self):
        """清空文件列表"""
        self.selected_files = []
        self.file_listbox.delete(0, tk.END)
        self.log("已清空文件列表")
        self.update_process_button()
    
    def toggle_output_dir(self):
        """切换输出目录状态"""
        if self.use_default_dir.get():
            self.output_dir_entry.config(state="normal")
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, DEFAULT_OUTPUT_DIR)
            self.output_dir_entry.config(state="readonly")
            self.output_dir = DEFAULT_OUTPUT_DIR
        else:
            self.output_dir_entry.config(state="normal")
    
    def select_output_dir(self):
        """选择输出目录"""
        if not self.use_default_dir.get():
            dir_path = filedialog.askdirectory(title="选择输出目录", initialdir=self.output_dir)
            if dir_path:
                self.output_dir = dir_path
                self.output_dir_entry.config(state="normal")
                self.output_dir_entry.delete(0, tk.END)
                self.output_dir_entry.insert(0, dir_path)
                self.output_dir_entry.config(state="readonly")
                self.log(f"输出目录已设置为: {dir_path}")
    
    def on_processing_mode_changed(self):
        """处理模式改变时的回调"""
        print(f"[DEBUG] on_processing_mode_changed 被调用，新模式: {self.processing_mode.get()}")
        if self.processing_mode.get() == "cascaded":
            # 级联模式：隐藏模型选择和说话人区分选项
            self.model_frame.grid_remove()
            self.speaker_frame.grid_remove()
            self.mode_info_label.config(
                text="级联模式：先用 Paraformer 做说话人区分，再用 SenseVoice 识别文本（保留情感标签）",
                foreground="green"
            )
            self.log("ℹ️ 已切换到级联模式", "info")
            self.log("   └─ 将使用 Paraformer 做说话人区分，SenseVoice 识别文本", "sub")
            
            # 清空模型，需要重新加载
            print("[DEBUG] 清空现有模型，准备重新加载")
            self.model = None
            self.paraformer_model = None
            self.sensevoice_model = None
            self.update_process_button()
            
            # 重新加载模型（级联模式）
            print("[DEBUG] 开始重新加载模型（级联模式）")
            self.status_label.config(text="正在加载模型...", foreground="orange")
            self.load_model_async()
        else:
            # 直接模式：显示模型选择和说话人区分选项
            self.model_frame.grid()
            self.speaker_frame.grid()
            self.mode_info_label.config(
                text="直接模式：使用单一模型进行识别",
                foreground="gray"
            )
            self.log("ℹ️ 已切换到直接模式", "info")
            self.on_model_type_changed()
    
    def on_model_type_changed(self):
        """模型类型改变时的回调（仅在直接模式下有效）"""
        if self.processing_mode.get() == "cascaded":
            return  # 级联模式下忽略此回调
        
        if self.model_type.get() == "paraformer":
            # Paraformer 支持 speaker diarization
            self.speaker_check.config(state="normal")
            self.enable_speaker.set(False)  # 默认不启用，让用户选择
            self.speaker_info_label.config(
                text="✅ Paraformer 模型支持说话人区分功能",
                foreground="green"
            )
        else:
            # SenseVoice 不支持 speaker diarization
            self.speaker_check.config(state="disabled")
            self.enable_speaker.set(False)
            self.speaker_info_label.config(
                text="⚠️ SenseVoice 模型不支持 timestamp，说话人识别功能不可用",
                foreground="orange"
            )
        
        # 如果模型已加载，需要重新加载
        if self.model is not None:
            self.log("ℹ️ 模型类型已更改，请重新加载模型", "info")
            self.model = None
            self.update_process_button()
    
    def on_speaker_changed(self):
        """说话人识别选项改变时的回调"""
        if self.enable_speaker.get() and self.model_type.get() == "sensevoice":
            # 不应该发生，但作为安全检查
            self.enable_speaker.set(False)
            messagebox.showwarning("警告", "SenseVoice 模型不支持说话人识别功能")
        
        # 如果模型已加载，需要重新加载
        if self.model is not None:
            self.log("ℹ️ 说话人识别设置已更改，请重新加载模型", "info")
            self.model = None
            self.update_process_button()
    
    def update_process_button(self):
        """更新处理按钮状态"""
        # 检查模型是否已加载（根据处理模式）
        model_ready = False
        if self.processing_mode.get() == "cascaded":
            model_ready = (self.paraformer_model is not None and 
                          self.sensevoice_model is not None)
            print(f"[DEBUG] update_process_button - 级联模式: paraformer={self.paraformer_model is not None}, sensevoice={self.sensevoice_model is not None}, model_ready={model_ready}")
        else:
            model_ready = (self.model is not None)
            print(f"[DEBUG] update_process_button - 直接模式: model={self.model is not None}, model_ready={model_ready}")
        
        print(f"[DEBUG] update_process_button - selected_files={len(self.selected_files) if self.selected_files else 0}, is_processing={self.is_processing}")
        print(f"[DEBUG] update_process_button - 按钮当前状态: {self.process_btn.cget('state')}")
        
        if model_ready and self.selected_files and not self.is_processing:
            print("[DEBUG] 启用开始处理按钮")
            self.process_btn.config(state="normal")
            print(f"[DEBUG] 按钮状态已设置为: {self.process_btn.cget('state')}")
            # 验证按钮是否真的被启用了
            actual_state = self.process_btn.cget('state')
            if actual_state != 'normal':
                print(f"[DEBUG] ⚠️ 警告：按钮状态设置失败！期望: normal, 实际: {actual_state}")
            self.stop_btn.config(state="disabled")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
        elif self.is_processing:
            self.process_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            if self.is_paused:
                self.pause_btn.config(state="disabled")
                self.resume_btn.config(state="normal")
            else:
                self.pause_btn.config(state="normal")
                self.resume_btn.config(state="disabled")
        else:
            print(f"[DEBUG] 禁用开始处理按钮 - model_ready={model_ready}, selected_files={len(self.selected_files) if self.selected_files else 0}, is_processing={self.is_processing}")
            self.process_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
    
    def stop_processing(self):
        """停止处理"""
        if self.is_processing:
            self.should_stop = True
            self.is_paused = False
            self.log("⏹️ 正在停止处理...", "error")
            self.status_label.config(text="正在停止...", foreground="orange")
    
    def pause_processing(self):
        """暂停处理"""
        if self.is_processing and not self.is_paused:
            self.is_paused = True
            self.log("⏸️ 处理已暂停", "warning")
            self.status_label.config(text="已暂停", foreground="orange")
            self.update_process_button()
    
    def resume_processing(self):
        """继续处理"""
        if self.is_processing and self.is_paused:
            self.is_paused = False
            self.log("▶️ 处理已继续", "info")
            self.status_label.config(text="处理中...", foreground="blue")
            self.update_process_button()
    
    def load_model_async(self):
        """异步加载模型"""
        def load():
            try:
                print(f"[DEBUG] load_model_async - processing_mode: {self.processing_mode.get()}")
                if self.processing_mode.get() == "cascaded":
                    # 级联模式：加载两个模型
                    print("[DEBUG] 级联模式：开始加载 Paraformer + Cam++ 模型")
                    self.log("🔄 正在加载 Paraformer + Cam++ 模型...")
                    start_time = time.time()
                    
                    paraformer_model = AutoModel(
                        model="paraformer-zh",
                        vad_model="fsmn-vad",
                        punc_model="ct-punc",
                        spk_model="cam++",
                        device=DEVICE,
                        ncpu=THREADS,
                        disable_update=True
                    )
                    
                    elapsed = time.time() - start_time
                    self.log(f"✅ Paraformer + Cam++ 模型加载完成，耗时: {elapsed:.2f}秒")
                    print(f"[DEBUG] Paraformer 模型加载完成，耗时: {elapsed:.2f}秒")
                    
                    print("[DEBUG] 开始加载 SenseVoice 模型")
                    self.log("🔄 正在加载 SenseVoice 模型...")
                    start_time = time.time()
                    
                    sensevoice_model = AutoModel(
                        model="iic/SenseVoiceSmall",
                        trust_remote_code=True,
                        vad_model="fsmn-vad",
                        vad_kwargs={"max_single_segment_time": 30000},
                        punc_model="ct-punc",
                        device=DEVICE,
                        ncpu=THREADS,
                        disable_update=True
                    )
                    
                    elapsed = time.time() - start_time
                    self.log(f"✅ SenseVoice 模型加载完成，耗时: {elapsed:.2f}秒")
                    print(f"[DEBUG] SenseVoice 模型加载完成，耗时: {elapsed:.2f}秒")
                    
                    self.paraformer_model = paraformer_model
                    self.sensevoice_model = sensevoice_model
                    self.model = None  # 直接模式下使用的模型
                    print(f"[DEBUG] 模型赋值完成: paraformer={self.paraformer_model is not None}, sensevoice={self.sensevoice_model is not None}")
                    
                else:
                    # 直接模式：加载单一模型
                    # 根据选择的模型类型确定模型 ID
                    if self.model_type.get() == "paraformer":
                        model_id = PARAFORMER_MODEL
                        model_name = "Paraformer"
                    else:
                        model_id = SENSEVOICE_MODEL
                        model_name = "SenseVoice"
                    
                    self.log(f"🔄 正在初始化模型: {model_name}...")
                    start_time = time.time()
                    
                    model_kwargs = {
                        "model": model_id,
                        "trust_remote_code": True,
                        "vad_model": "fsmn-vad",
                        "vad_kwargs": {"max_single_segment_time": 30000},
                        "device": DEVICE,
                        "ncpu": THREADS,
                        "disable_update": True,
                        "punc_model": "ct-punc"  # 显式指定 punc_model
                    }
                    
                    # 如果启用说话人区分（仅 Paraformer 支持）
                    if self.enable_speaker.get() and self.model_type.get() == "paraformer":
                        model_kwargs["spk_model"] = SPK_MODEL
                        self.log("📢 已启用说话人区分功能")
                        self.log("   └─ 已自动加载标点符号模型（说话人识别需要）", "sub")
                        self.log("   ℹ️ 输出时将过滤掉 timestamp，只显示说话人 ID 和文本", "info")
                    elif self.enable_speaker.get() and self.model_type.get() == "sensevoice":
                        # 不应该发生，但作为安全检查
                        self.log("⚠️ SenseVoice 模型不支持说话人识别，已自动禁用", "warning")
                        self.enable_speaker.set(False)
                    
                    self.model = AutoModel(**model_kwargs)
                    self.paraformer_model = None
                    self.sensevoice_model = None
                    
                    elapsed = time.time() - start_time
                    self.log(f"✅ 模型加载完成，耗时: {elapsed:.2f}秒")
                
                self.status_label.config(text="就绪", foreground="green")
                print("[DEBUG] 调用 update_process_button 更新按钮状态")
                self.update_process_button()
                print("[DEBUG] update_process_button 调用完成")
            except Exception as e:
                error_traceback = traceback.format_exc()
                self.log(f"模型加载失败: {str(e)}", "error")
                self.log(f"错误类型: {type(e).__name__}", "error")
                
                # 在详细日志中记录完整错误
                self.log_detail("模型加载失败", "error")
                self.log_detail(f"错误类型: {type(e).__name__}", "error")
                self.log_detail(f"错误信息: {str(e)}", "error")
                self.log_detail("完整错误堆栈:", "error")
                self.log_detail(error_traceback, "error")
                
                self.status_label.config(text="模型加载失败", foreground="red")
                messagebox.showerror("错误", f"模型加载失败:\n{str(e)}\n\n详细错误信息请查看「详细日志」标签页")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def process_audio(self, audio_file):
        """处理单个音频文件"""
        if not os.path.exists(audio_file):
            return None, f"文件不存在: {audio_file}"
        
        filename = os.path.basename(audio_file)
        self.log(f"🎙️ 开始处理: {filename}")
        
        # 获取文件大小用于显示
        try:
            file_size = os.path.getsize(audio_file)
            file_size_mb = file_size / (1024 * 1024)
            self.log(f"文件大小: {file_size_mb:.2f} MB", "sub")
        except:
            pass
        
        # 根据处理模式选择不同的处理方式
        if self.processing_mode.get() == "cascaded":
            return self.process_audio_cascaded(audio_file)
        else:
            return self.process_audio_direct(audio_file)
    
    def process_audio_cascaded(self, audio_file):
        """级联模式处理音频：先 Paraformer 做 diarization，再用 SenseVoice 识别"""
        if not self.paraformer_model or not self.sensevoice_model:
            return None, "级联模式需要加载 Paraformer 和 SenseVoice 模型"
        
        start_time = time.time()
        
        try:
            # 导入级联系统模块
            from run_cascaded_system import (
                process_audio_cascaded as cascaded_process,
                format_cascaded_result
            )
            
            self.log("="*60)
            self.log("🔄 步骤 1/3: 使用 Paraformer 进行说话人区分...")
            self.log("="*60)
            
            # 调用级联处理函数（传入日志回调函数）
            final_results = cascaded_process(
                audio_file, 
                self.paraformer_model, 
                self.sensevoice_model,
                log_callback=self.log,
                log_detail_callback=self.log_detail
            )
            
            total_time = time.time() - start_time
            self.log(f"✅ 级联处理完成，总耗时: {total_time:.2f}秒")
            
            # 格式化结果
            formatted_result = format_cascaded_result(final_results, audio_file)
            
            # 返回结果（格式与直接模式保持一致）
            # 将级联结果转换为直接模式的格式
            result_dict = {
                "text": formatted_result,
                "speaker": final_results,  # 包含说话人信息的列表
                "raw": final_results
            }
            
            return result_dict, None
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"级联处理失败: {str(e)}"
            
            self.log(f"❌ {error_msg}", "error")
            self.log_detail(f"级联处理音频文件时发生错误: {audio_file}", "error")
            self.log_detail(f"错误类型: {type(e).__name__}", "error")
            self.log_detail(f"错误信息: {str(e)}", "error")
            self.log_detail("完整错误堆栈:", "error")
            self.log_detail(error_traceback, "error")
            self.log_notebook.select(1)  # 切换到详细日志标签页
            
            return None, error_msg
    
    def process_audio_direct(self, audio_file):
        """直接模式处理音频：使用单一模型"""
        if not self.model:
            return None, "模型未加载"
        
        start_time = time.time()
        
        try:
            # 显示处理步骤（由于 model.generate 是黑盒，我们只能显示整体进度）
            self.log("开始推理处理 (VAD + ASR + 后处理)...", "sub")
            
            # 根据模型类型和说话人识别设置准备参数
            generate_kwargs = {
                "input": audio_file,
                "cache": {},
                "language": "auto",
                "use_itn": True,
                "batch_size_s": 60,
                "merge_vad": True,
            }
            
            # Paraformer 模型支持 speaker diarization，不需要特殊设置
            # SenseVoice 模型不支持，如果启用会报错（已在模型加载时检查）
            if self.enable_speaker.get() and self.model_type.get() == "paraformer":
                self.log_detail(f"使用 Paraformer 模型进行说话人识别", "info")
            
            try:
                res = self.model.generate(**generate_kwargs)
            except (KeyError, Exception) as e:
                # 检查是否是 timestamp 相关的错误
                error_str = str(e)
                error_type = type(e).__name__
                
                # 检查是否是 timestamp 错误（更全面的检测）
                is_timestamp_error = (
                    "'timestamp'" in error_str or 
                    '"timestamp"' in error_str or
                    "timestamp" in error_str.lower() or
                    (error_type == "KeyError" and ("timestamp" in error_str or "timestamp" in str(e.args)))
                )
                
                # 检查错误堆栈中是否包含 timestamp
                import traceback
                tb_str = traceback.format_exc()
                if "timestamp" in tb_str.lower():
                    is_timestamp_error = True
                
                if is_timestamp_error:
                    # 检测到 timestamp 错误
                    self.log("⚠️ 检测到 timestamp 错误", "warning")
                    self.log("⚠️ SenseVoice 模型不支持说话人识别所需的 timestamp 字段", "warning")
                    self.log("⚠️ 说话人识别功能无法使用", "warning")
                    self.log_detail(f"错误类型: {error_type}", "error")
                    self.log_detail(f"错误信息: {error_str}", "error")
                    self.log_detail("SenseVoice 模型不支持生成 timestamp，说话人识别功能需要 timestamp", "error")
                    
                    # 检查模型是否加载了 spk_model（通过检查模型属性）
                    has_spk_model = hasattr(self.model, 'spk_model') and self.model.spk_model is not None
                    if has_spk_model:
                        self.log_detail("由于模型已加载 spk_model，需要重新加载模型才能完全禁用说话人识别", "warning")
                        # 自动禁用说话人识别复选框
                        self.enable_speaker.set(False)
                        self.log("ℹ️ 已自动禁用「启用说话人区分」选项", "info")
                        self.log("ℹ️ 请重新启动程序以重新加载模型（不加载 spk_model）", "info")
                        self.log_detail("已自动禁用说话人识别功能，建议重新启动程序", "info")
                    
                    # 返回友好的错误信息，不抛出异常，让处理流程继续
                    error_msg = (
                        f"说话人识别功能无法使用：SenseVoice 模型不支持 timestamp。\n"
                    )
                    if has_spk_model:
                        error_msg += (
                            f"已自动禁用「启用说话人区分」选项。\n"
                            f"请重新启动程序以重新加载模型（不加载 spk_model），然后重新处理文件。"
                        )
                    else:
                        error_msg += "请禁用说话人识别功能后重新处理文件。"
                    
                    return None, error_msg
                else:
                    # 其他错误，直接抛出
                    raise
            
            inference_time = time.time() - start_time
            
            # 显示处理完成信息
            if inference_time > 1.0:  # 如果耗时超过1秒，显示详细信息
                self.log(f"推理完成，耗时: {inference_time:.2f}秒", "sub")
            
            if res:
                # 处理结果，保留说话人信息
                if isinstance(res, list):
                    if len(res) > 0:
                        result_item = res[0]
                        if isinstance(result_item, dict):
                            text = rich_transcription_postprocess(result_item.get("text", ""))
                            # 移除 SenseVoice 标签和 emoji
                            text = self.remove_sensevoice_tags(text)
                            text = self.remove_emoji(text)
                            speaker_info = result_item.get("spk", None)
                            if speaker_info:
                                self.log(f"📢 检测到说话人信息", "sub")
                            self.log(f"✅ 处理完成，总耗时: {inference_time:.2f}秒")
                            return {"text": text, "speaker": speaker_info, "raw": result_item}, None
                        else:
                            text = rich_transcription_postprocess(result_item if result_item else "")
                            # 移除 SenseVoice 标签和 emoji
                            text = self.remove_sensevoice_tags(text)
                            text = self.remove_emoji(text)
                            self.log(f"✅ 处理完成，总耗时: {inference_time:.2f}秒")
                            return {"text": text, "speaker": None, "raw": result_item}, None
                    else:
                        return None, "未检测到有效语音"
                elif isinstance(res, dict):
                    text = rich_transcription_postprocess(res.get("text", ""))
                    # 移除 SenseVoice 标签和 emoji
                    text = self.remove_sensevoice_tags(text)
                    text = self.remove_emoji(text)
                    speaker_info = res.get("spk", None)
                    if speaker_info:
                        self.log(f"📢 检测到说话人信息", "sub")
                    self.log(f"✅ 处理完成，总耗时: {inference_time:.2f}秒")
                    return {"text": text, "speaker": speaker_info, "raw": res}, None
                else:
                    text = str(res)
                    # 移除 emoji
                    text = self.remove_emoji(text)
                    self.log(f"✅ 处理完成，总耗时: {inference_time:.2f}秒")
                    return {"text": text, "speaker": None, "raw": res}, None
            else:
                return None, "未检测到有效语音"
        except Exception as e:
            # 获取完整的错误堆栈信息
            error_msg = f"处理失败: {str(e)}"
            error_traceback = traceback.format_exc()
            
            # 在主日志中显示简要错误信息
            self.log(f"错误类型: {type(e).__name__}", "error")
            self.log(f"错误信息: {str(e)}", "error")
            
            # 在详细日志中显示完整堆栈
            self.log_detail(f"处理音频文件时发生错误: {audio_file}", "error")
            self.log_detail(f"错误类型: {type(e).__name__}", "error")
            self.log_detail(f"错误信息: {str(e)}", "error")
            self.log_detail("完整错误堆栈:", "error")
            self.log_detail(error_traceback, "error")
            
            return None, error_msg
    
    def start_processing(self):
        """开始处理"""
        print("[DEBUG] start_processing 被调用")
        print(f"[DEBUG] selected_files: {self.selected_files}")
        print(f"[DEBUG] processing_mode: {self.processing_mode.get()}")
        print(f"[DEBUG] model: {self.model}")
        print(f"[DEBUG] paraformer_model: {self.paraformer_model}")
        print(f"[DEBUG] sensevoice_model: {self.sensevoice_model}")
        print(f"[DEBUG] is_processing: {self.is_processing}")
        
        if not self.selected_files:
            print("[DEBUG] 没有选择文件")
            messagebox.showwarning("警告", "请先选择音频文件")
            return
        
        if not self.use_default_dir.get() and not self.output_dir:
            print("[DEBUG] 没有选择输出目录")
            messagebox.showwarning("警告", "请选择输出目录")
            return
        
        # 检查模型是否已加载
        if self.processing_mode.get() == "cascaded":
            if not self.paraformer_model or not self.sensevoice_model:
                print("[DEBUG] 级联模式：模型未完全加载")
                print(f"[DEBUG] paraformer_model is None: {self.paraformer_model is None}")
                print(f"[DEBUG] sensevoice_model is None: {self.sensevoice_model is None}")
                messagebox.showerror("错误", "级联模式需要加载 Paraformer 和 SenseVoice 模型，请等待模型加载完成")
                return
        else:
            if not self.model:
                print("[DEBUG] 直接模式：模型未加载")
                messagebox.showerror("错误", "模型未加载，请等待模型加载完成")
                return
        
        print("[DEBUG] 开始处理流程")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 重置状态
        self.is_processing = True
        self.is_paused = False
        self.should_stop = False
        print("[DEBUG] 更新按钮状态")
        self.update_process_button()
        self.status_label.config(text="处理中...", foreground="blue")
        self.progress_var.set(0)
        
        def heartbeat():
            """心跳线程，定期更新状态显示程序仍在运行"""
            while not self.heartbeat_stop.is_set():
                if self.is_processing and self.processing_start_time:
                    elapsed = time.time() - self.processing_start_time
                    # 每3秒更新一次心跳（不会频繁更新影响性能）
                    self.heartbeat_stop.wait(3)
                    if self.is_processing:
                        self.status_label.config(
                            text=f"处理中... (已运行 {int(elapsed)}秒)",
                            foreground="blue"
                        )
                else:
                    break
        
        def process():
            try:
                print("[DEBUG] process 线程开始执行")
                total_files = len(self.selected_files)
                print(f"[DEBUG] 总共 {total_files} 个文件需要处理")
                self.log(f"\n{'='*60}")
                self.log(f"开始处理 {total_files} 个文件")
                self.log(f"{'='*60}\n")
                
                success_count = 0
                fail_count = 0
                
                # 启动心跳线程
                print("[DEBUG] 启动心跳线程")
                self.heartbeat_stop.clear()
                self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                self.heartbeat_thread.start()
                print("[DEBUG] 心跳线程已启动")
                
                for idx, audio_file in enumerate(self.selected_files, 1):
                    # 检查是否应该停止
                    if self.should_stop:
                        self.log("⏹️ 处理已停止", "error")
                        break
                    
                    # 等待暂停状态解除
                    while self.is_paused and not self.should_stop:
                        time.sleep(0.5)  # 每0.5秒检查一次
                    
                    # 再次检查是否应该停止（可能在暂停期间被停止）
                    if self.should_stop:
                        self.log("⏹️ 处理已停止", "error")
                        break
                    
                    self.processing_file_index = idx
                    self.processing_start_time = time.time()
                    
                    # 更新进度
                    progress = (idx - 1) / total_files * 100
                    self.progress_var.set(progress)
                    
                    self.log(f"\n文件 {idx}/{total_files}: {os.path.basename(audio_file)}")
                    self.log("-" * 60)
                    
                    result, error = self.process_audio(audio_file)
                    
                    # 检查是否在处理过程中被停止
                    if self.should_stop:
                        self.log("⏹️ 处理已停止", "error")
                        break
                    
                    if result:
                        try:
                            # 保存结果
                            base_name = os.path.splitext(os.path.basename(audio_file))[0]
                            output_file = os.path.join(self.output_dir, f"{base_name}_transcription.txt")
                            
                            # 格式化结果（包含说话人信息）
                            formatted_result = self.format_result_with_speaker(result, audio_file)
                            
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(formatted_result)
                            
                            self.log(f"💾 结果已保存到: {output_file}")
                            self.log("\n识别结果:")
                            self.log(formatted_result)
                            success_count += 1
                        except Exception as format_error:
                            # 格式化或保存结果时出错
                            error_msg = f"保存结果时出错: {str(format_error)}"
                            error_traceback = traceback.format_exc()
                            
                            # 在主日志中显示简要错误
                            self.log(error_msg, "error")
                            self.log(f"错误类型: {type(format_error).__name__}", "error")
                            
                            # 在详细日志中显示完整堆栈
                            self.log_detail(f"保存结果时出错: {audio_file}", "error")
                            self.log_detail(f"错误类型: {type(format_error).__name__}", "error")
                            self.log_detail(f"错误信息: {str(format_error)}", "error")
                            self.log_detail("完整错误堆栈:", "error")
                            self.log_detail(error_traceback, "error")
                            
                            fail_count += 1
                    else:
                        # 处理失败，error 已经包含了详细错误信息（在 process_audio 中已记录）
                        if error:
                            self.log(f"❌ {error}")
                        else:
                            self.log(f"❌ 处理失败: 未知错误")
                        fail_count += 1
                
                # 停止心跳线程
                self.heartbeat_stop.set()
                
                # 完成
                if self.should_stop:
                    self.log(f"\n{'='*60}")
                    self.log(f"⏹️ 处理已停止！已处理: {success_count + fail_count}/{total_files}, 成功: {success_count}, 失败: {fail_count}")
                    self.log(f"{'='*60}\n")
                    self.status_label.config(text=f"已停止 (已处理: {success_count + fail_count}/{total_files})", foreground="orange")
                    messagebox.showinfo("已停止", f"处理已停止！\n已处理: {success_count + fail_count}/{total_files}\n成功: {success_count}\n失败: {fail_count}")
                else:
                    self.progress_var.set(100)
                    self.log(f"\n{'='*60}")
                    self.log(f"✅ 处理完成！成功: {success_count}, 失败: {fail_count}")
                    self.log(f"{'='*60}\n")
                    self.status_label.config(text=f"完成 (成功: {success_count}, 失败: {fail_count})", foreground="green")
                    messagebox.showinfo("完成", f"处理完成！\n成功: {success_count}\n失败: {fail_count}")
                
            except Exception as e:
                self.heartbeat_stop.set()
                error_msg = f"处理过程中出错: {str(e)}"
                error_traceback = traceback.format_exc()
                
                # 在主日志中显示简要错误
                self.log(error_msg, "error")
                self.log(f"错误类型: {type(e).__name__}", "error")
                
                # 在详细日志中显示完整堆栈
                self.log_detail("处理过程中发生未捕获的异常", "error")
                self.log_detail(f"错误类型: {type(e).__name__}", "error")
                self.log_detail(f"错误信息: {str(e)}", "error")
                self.log_detail("完整错误堆栈:", "error")
                self.log_detail(error_traceback, "error")
                
                self.status_label.config(text="处理失败", foreground="red")
                messagebox.showerror("错误", f"处理过程中出错:\n{str(e)}\n\n详细错误信息请查看「详细日志」标签页")
            finally:
                self.is_processing = False
                self.is_paused = False
                self.should_stop = False
                self.processing_start_time = None
                self.processing_thread = None
                self.update_process_button()
        
        print("[DEBUG] 创建处理线程")
        self.processing_thread = threading.Thread(target=process, daemon=True)
        print("[DEBUG] 启动处理线程")
        self.processing_thread.start()
        print(f"[DEBUG] 处理线程已启动，线程ID: {self.processing_thread.ident}")
        print(f"[DEBUG] 线程是否存活: {self.processing_thread.is_alive()}")
    
    def remove_sensevoice_tags(self, text):
        """
        移除 SenseVoice 输出的标签，只保留纯文本
        
        移除的标签格式：
        - <|en|>, <|zh|>, <|yue|>, <|ja|> 等语言标签
        - <|NEUTRAL|>, <|EMO_UNKNOWN|> 等情绪标签
        - <|Speech|>, <|within|> 等其他标签
        """
        if not text:
            return ""
        
        # 移除所有 <|...|> 格式的标签
        tag_pattern = re.compile(r'<\|[^|]+\|>')
        text = tag_pattern.sub('', text)
        
        # 清理多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_emoji(self, text):
        """移除文本中的 emoji，保留标点符号和基本字符（包括中文）"""
        # 移除 emoji（Unicode emoji 范围）
        # 保留：字母、数字、中文、标点符号、空格
        # 移除：emoji、特殊符号等
        # 注意：移除了 \U000024C2-\U0001F251 范围，因为它包含了中文字符范围
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号和象形文字
            "\U0001F680-\U0001F6FF"  # 交通和地图符号
            "\U0001F1E0-\U0001F1FF"  # 旗帜
            "\U00002702-\U000027B0"  # 其他符号
            "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
            "\U0001FA00-\U0001FA6F"  # 扩展符号
            "\U0001FA70-\U0001FAFF"  # 扩展符号
            "\U00002600-\U000026FF"  # 杂项符号
            "\U00002700-\U000027BF"  # 装饰符号
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text).strip()
    
    def format_result_with_speaker(self, result, audio_file):
        """格式化带说话人信息的结果"""
        try:
            if not result:
                return "未检测到有效语音"
            
            # 检查是否是级联模式的结果（speaker 是列表，包含 spk_id, start, end, text）
            is_cascaded_result = False
            speaker_info = result.get("speaker", None) if isinstance(result, dict) else None
            
            if isinstance(speaker_info, list) and len(speaker_info) > 0:
                # 检查第一个元素是否包含级联模式的特征字段
                first_item = speaker_info[0]
                if isinstance(first_item, dict) and "spk_id" in first_item:
                    is_cascaded_result = True
            
            if is_cascaded_result:
                # 级联模式的结果格式
                output_lines = []
                output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
                output_lines.append("="*60 + "\n")
                output_lines.append("📢 说话人区分结果（使用 SenseVoice 识别）:\n")
                output_lines.append("-"*60 + "\n")
                
                for item in speaker_info:
                    if isinstance(item, dict):
                        spk_id = item.get("spk_id", "Unknown")
                        text = item.get("text", "")
                        # 移除 emoji
                        text = self.remove_emoji(text)
                        output_lines.append(f"说话人 {spk_id}: {text}\n")
                
                output_lines.append("\n" + "="*60 + "\n")
                return "".join(output_lines)
            
            # 直接模式的结果格式（原有逻辑）
            text = result.get("text", "") if isinstance(result, dict) else result
            raw_data = result.get("raw", {}) if isinstance(result, dict) else {}
            
            # 移除 emoji
            text = self.remove_emoji(text)
            
            output_lines = []
            output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
            output_lines.append("="*60 + "\n")
            
            # 如果有说话人信息，格式化输出（过滤掉 timestamp）
            if speaker_info:
                output_lines.append("📢 说话人区分结果:\n")
                output_lines.append("-"*60 + "\n")
                
                try:
                    if isinstance(speaker_info, list):
                        for idx, spk in enumerate(speaker_info):
                            try:
                                if isinstance(spk, dict):
                                    spk_id = spk.get("spk_id", f"Speaker_{idx}")
                                    # 提取文本，不显示 timestamp
                                    spk_text = spk.get("text", "") or spk.get("sentence", "")
                                    if spk_text:
                                        output_lines.append(f"说话人 {spk_id}: {spk_text}\n")
                                    else:
                                        output_lines.append(f"说话人 {spk_id}:\n")
                                else:
                                    output_lines.append(f"说话人 {idx}: {spk}\n")
                            except Exception as e:
                                output_lines.append(f"说话人 {idx}: [格式化错误: {str(e)}]\n")
                    elif isinstance(speaker_info, dict):
                        for spk_id, info in speaker_info.items():
                            # 如果 info 是字典，提取文本
                            if isinstance(info, dict):
                                info_text = info.get("text", "") or info.get("sentence", "")
                                output_lines.append(f"说话人 {spk_id}: {info_text}\n")
                            else:
                                output_lines.append(f"说话人 {spk_id}: {info}\n")
                    else:
                        output_lines.append(f"说话人信息: {speaker_info}\n")
                except Exception as e:
                    output_lines.append(f"[说话人信息格式化错误: {str(e)}]\n")
                
                output_lines.append("\n")
            
            # 转录文本（已移除 emoji）
            output_lines.append("识别结果:\n")
            output_lines.append("-"*60 + "\n")
            output_lines.append(text + "\n")
            
            # 不输出 timestamp 信息（根据用户需求）
            # 如果需要 timestamp，可以取消下面的注释
            # try:
            #     if isinstance(raw_data, dict):
            #         timestamp = raw_data.get("timestamp", None)
            #         if timestamp:
            #             output_lines.append("\n时间戳信息:\n")
            #             output_lines.append(f"{timestamp}\n")
            # except Exception as e:
            #     pass
            
            return "".join(output_lines)
        except Exception as e:
            # 格式化失败，记录详细错误并返回基本信息
            error_traceback = traceback.format_exc()
            error_msg = f"格式化结果时出错: {str(e)}"
            
            # 在详细日志中记录完整错误
            self.log_detail(f"格式化结果时出错: {audio_file}", "error")
            self.log_detail(f"错误类型: {type(e).__name__}", "error")
            self.log_detail(f"错误信息: {str(e)}", "error")
            self.log_detail(f"原始结果类型: {type(result)}", "debug")
            if isinstance(result, dict):
                self.log_detail(f"结果键: {list(result.keys())}", "debug")
            self.log_detail("完整错误堆栈:", "error")
            self.log_detail(error_traceback, "error")
            
            # 返回简化的错误信息
            return f"音频文件: {os.path.basename(audio_file)}\n\n错误: {error_msg}\n\n详细错误信息请查看「详细日志」标签页"

def main():
    root = tk.Tk()
    app = AudioTranscriptionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

