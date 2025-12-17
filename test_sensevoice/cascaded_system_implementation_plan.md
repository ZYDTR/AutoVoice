# 级联系统实现计划：先 Diarization 再用 SenseVoice

## 一、调研结果总结

### 1.1 Paraformer + Cam++ 输出格式确认

经过实际测试，确认了 Paraformer + Cam++ 的输出结构：

```python
result = [
    {
        'key': '文件名',
        'text': '完整转录文本',
        'timestamp': [[start_ms, end_ms], ...],  # 时间戳列表
        'sentence_info': [
            {
                'text': '句子文本',
                'start': 2990,      # 句子开始时间（毫秒）
                'end': 7990,        # 句子结束时间（毫秒）
                'timestamp': [[2990, 3230], ...],  # 句子内的时间戳
                'spk': 1            # 说话人 ID（整数）
            },
            ...
        ]
    }
]
```

**关键发现：**
- ✅ `sentence_info` 中包含每个句子的 `start`、`end` 和 `spk`（说话人ID）
- ✅ 时间戳单位是**毫秒**
- ✅ 说话人ID是整数（如 1, 2, 3...）

### 1.2 SenseVoice 输出格式

SenseVoice 的输出格式（已知）：
```python
result = [
    {
        'text': '转录文本（包含情感标签）',
        # 不包含 timestamp 和 spk
    }
]
```

**关键特性：**
- ✅ 包含情感标签（如 `<|happy|>`、`<|music|>`）
- ✅ 识别准确率高
- ❌ 不支持 timestamp 和 speaker diarization

## 二、实现方案设计

### 2.1 核心思路

**"用 Paraformer 定位定人，用 SenseVoice 修正内容"**

1. **第一步（定位与定人）**：
   - 使用 Paraformer + Cam++ 处理完整音频
   - 获取每个句子的时间戳（start, end）和说话人ID（spk）
   - 忽略 Paraformer 的文本输出（因为要用 SenseVoice 重新识别）

2. **第二步（识别内容）**：
   - 根据时间戳提取音频片段
   - 使用 SenseVoice 重新识别每个片段
   - 获取带情感标签的文本

3. **第三步（合并结果）**：
   - 将说话人ID（来自Paraformer）与文本（来自SenseVoice）合并
   - 输出最终结果

### 2.2 技术实现要点

#### 2.2.1 音频片段提取

需要使用音频处理库提取指定时间段的音频：

**方案A：使用 soundfile + numpy（推荐）**
```python
import soundfile as sf
import numpy as np

def extract_audio_segment(audio_path, start_ms, end_ms):
    """
    提取音频片段
    
    Args:
        audio_path: 音频文件路径
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
    
    Returns:
        audio_data: 音频数据（numpy array）
        sample_rate: 采样率
    """
    # 读取完整音频
    audio_data, sample_rate = sf.read(audio_path)
    
    # 转换为采样点索引
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    
    # 提取片段
    segment = audio_data[start_sample:end_sample]
    
    return segment, sample_rate
```

**方案B：使用 pydub（更简单，但需要 ffmpeg）**
```python
from pydub import AudioSegment

def extract_audio_segment_pydub(audio_path, start_ms, end_ms):
    """
    使用 pydub 提取音频片段
    """
    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_ms:end_ms]
    
    # 转换为 numpy array（如果需要）
    import numpy as np
    samples = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
    
    return samples, segment.frame_rate
```

**推荐使用方案A（soundfile）**，因为：
- ✅ 不需要额外依赖（ffmpeg）
- ✅ 性能更好
- ✅ 支持更多音频格式

#### 2.2.2 临时文件处理

SenseVoice 的 `generate` 方法需要文件路径或 numpy array。如果传入 numpy array，需要确保格式正确。

**方案：使用临时文件或直接传入 numpy array**
```python
import tempfile
import os

def process_segment_with_sensevoice(sense_model, audio_segment, sample_rate):
    """
    使用 SenseVoice 处理音频片段
    
    方案1：保存为临时文件
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_segment, sample_rate)
        result = sense_model.generate(input=tmp_file.name, ...)
        os.unlink(tmp_file.name)  # 删除临时文件
        return result
    
    # 方案2：直接传入 numpy array（如果支持）
    # result = sense_model.generate(input=audio_segment, ...)
```

#### 2.2.3 结果合并

```python
def merge_results(paraformer_result, sensevoice_results):
    """
    合并 Paraformer 的说话人信息和 SenseVoice 的文本
    
    Args:
        paraformer_result: Paraformer 的输出（包含 sentence_info）
        sensevoice_results: 每个片段对应的 SenseVoice 输出列表
    
    Returns:
        merged_results: 合并后的结果列表
    """
    merged = []
    sentence_info = paraformer_result['sentence_info']
    
    for i, sent_info in enumerate(sentence_info):
        # 获取说话人ID
        spk_id = sent_info['spk']
        
        # 获取 SenseVoice 识别的文本
        if i < len(sensevoice_results):
            sense_text = sensevoice_results[i]
        else:
            sense_text = sent_info['text']  # 降级使用 Paraformer 的文本
        
        merged.append({
            'spk_id': spk_id,
            'start': sent_info['start'],
            'end': sent_info['end'],
            'text': sense_text  # 来自 SenseVoice，包含情感标签
        })
    
    return merged
```

## 三、详细实现步骤

### 3.1 核心函数实现

```python
def process_audio_cascaded(audio_path, paraformer_model, sensevoice_model):
    """
    级联处理音频：先 Paraformer 做 diarization，再用 SenseVoice 识别
    
    Args:
        audio_path: 音频文件路径
        paraformer_model: Paraformer + Cam++ 模型
        sensevoice_model: SenseVoice 模型
    
    Returns:
        final_results: 最终结果列表
    """
    # === 步骤 1: Paraformer 处理（获取时间戳和说话人ID） ===
    print("🔄 步骤 1/3: 使用 Paraformer 进行说话人区分...")
    paraformer_res = paraformer_model.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )
    
    if not paraformer_res or len(paraformer_res) == 0:
        raise ValueError("Paraformer 处理失败")
    
    paraformer_result = paraformer_res[0]
    sentence_info = paraformer_result.get('sentence_info', [])
    
    if not sentence_info:
        raise ValueError("未检测到句子信息")
    
    print(f"✅ 检测到 {len(sentence_info)} 个句子片段")
    
    # === 步骤 2: 提取音频片段并用 SenseVoice 重新识别 ===
    print("🔄 步骤 2/3: 提取音频片段并用 SenseVoice 重新识别...")
    sensevoice_results = []
    
    for idx, sent_info in enumerate(sentence_info):
        start_ms = sent_info['start']
        end_ms = sent_info['end']
        
        print(f"  处理片段 {idx+1}/{len(sentence_info)}: {start_ms}ms - {end_ms}ms")
        
        # 提取音频片段
        audio_segment, sample_rate = extract_audio_segment(
            audio_path, start_ms, end_ms
        )
        
        # 使用 SenseVoice 识别
        sense_res = sensevoice_model.generate(
            input=audio_segment,  # 或临时文件路径
            cache={},
            language="auto",
            use_itn=True,
        )
        
        # 提取文本
        if sense_res and len(sense_res) > 0:
            if isinstance(sense_res[0], dict):
                text = sense_res[0].get('text', '')
            else:
                text = str(sense_res[0])
        else:
            text = sent_info['text']  # 降级使用 Paraformer 的文本
        
        sensevoice_results.append(text)
    
    # === 步骤 3: 合并结果 ===
    print("🔄 步骤 3/3: 合并结果...")
    final_results = merge_results(paraformer_result, sensevoice_results)
    
    return final_results
```

### 3.2 模型初始化

```python
def setup_cascaded_models():
    """
    初始化级联系统所需的模型
    """
    print("🔄 正在加载 Paraformer + Cam++ 模型...")
    paraformer_model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device="cpu",
        ncpu=4,
        disable_update=True
    )
    
    print("🔄 正在加载 SenseVoice 模型...")
    sensevoice_model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        device="cpu",
        ncpu=4,
        disable_update=True
    )
    
    return paraformer_model, sensevoice_model
```

### 3.3 输出格式化

```python
def format_cascaded_result(final_results, audio_file):
    """
    格式化级联系统的输出结果
    """
    output_lines = []
    output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
    output_lines.append("="*60 + "\n")
    output_lines.append("📢 说话人区分结果（使用 SenseVoice 识别）:\n")
    output_lines.append("-"*60 + "\n")
    
    for result in final_results:
        spk_id = result['spk_id']
        text = result['text']
        # 移除 emoji（如果需要）
        text = remove_emoji(text)
        
        output_lines.append(f"说话人 {spk_id}: {text}\n")
    
    return "".join(output_lines)
```

## 四、集成到现有系统

### 4.1 GUI 界面修改

在 GUI 中添加新的处理模式选项：

```python
# 在 create_widgets 中添加
processing_mode_frame = ttk.LabelFrame(main_frame, text="处理模式", padding="10")
processing_mode_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

self.processing_mode = tk.StringVar(value="direct")  # "direct" 或 "cascaded"

ttk.Radiobutton(
    processing_mode_frame,
    text="直接模式（单一模型）",
    variable=self.processing_mode,
    value="direct"
).grid(row=0, column=0, padx=10, sticky=tk.W)

ttk.Radiobutton(
    processing_mode_frame,
    text="级联模式（Paraformer + SenseVoice）",
    variable=self.processing_mode,
    value="cascaded"
).grid(row=0, column=1, padx=10, sticky=tk.W)

info_label = ttk.Label(
    processing_mode_frame,
    text="级联模式：先用 Paraformer 做说话人区分，再用 SenseVoice 识别文本（保留情感标签）",
    foreground="gray",
    font=("Arial", 9)
)
info_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
```

### 4.2 命令行脚本修改

添加命令行参数或配置选项：

```python
# 在配置区域添加
ENABLE_CASCADED_MODE = False  # 是否启用级联模式

# 在 process_audio 函数中添加模式判断
if ENABLE_CASCADED_MODE:
    result = process_audio_cascaded(audio_file, paraformer_model, sensevoice_model)
else:
    result = process_audio_direct(model, audio_file)
```

## 五、性能优化建议

### 5.1 批量处理优化

- **并行处理片段**：使用多线程/多进程并行处理多个音频片段
- **缓存机制**：缓存已加载的模型，避免重复加载

### 5.2 内存优化

- **流式处理**：对于长音频，可以流式处理，避免一次性加载所有片段
- **临时文件清理**：及时删除临时文件，释放磁盘空间

### 5.3 错误处理

- **降级策略**：如果 SenseVoice 识别失败，降级使用 Paraformer 的文本
- **超时处理**：为每个片段设置超时时间，避免卡死

## 六、测试计划

### 6.1 单元测试

1. 测试音频片段提取函数
2. 测试结果合并函数
3. 测试输出格式化函数

### 6.2 集成测试

1. 测试完整的级联流程
2. 测试不同长度的音频文件
3. 测试多说话人场景

### 6.3 性能测试

1. 对比级联模式与直接模式的性能
2. 测试内存使用情况
3. 测试处理时间

## 七、实施优先级

### Phase 1: 核心功能实现（高优先级）
1. ✅ 实现音频片段提取函数
2. ✅ 实现级联处理核心逻辑
3. ✅ 实现结果合并和格式化

### Phase 2: 集成到现有系统（中优先级）
1. ✅ 更新命令行脚本
2. ✅ 更新 GUI 界面
3. ✅ 添加配置选项

### Phase 3: 优化和测试（低优先级）
1. ⏳ 性能优化
2. ⏳ 错误处理完善
3. ⏳ 文档和测试

## 八、已知问题和解决方案

### 8.1 音频格式兼容性

**问题**：不同音频格式可能需要不同的处理方式

**解决方案**：
- 使用 soundfile 统一处理（支持多种格式）
- 如果遇到不支持的格式，使用 ffmpeg 转换

### 8.2 SenseVoice 输入格式

**问题**：SenseVoice 的 `generate` 方法可能不支持直接传入 numpy array

**解决方案**：
- 先测试是否支持 numpy array
- 如果不支持，使用临时文件

### 8.3 时间戳精度

**问题**：毫秒级时间戳可能存在精度误差

**解决方案**：
- 在提取片段时添加前后缓冲（如前后各 100ms）
- 确保提取的片段包含完整的语音内容

## 九、参考资料

1. FunASR 官方文档：https://github.com/modelscope/FunASR
2. soundfile 文档：https://pysoundfile.readthedocs.io/
3. Paraformer 模型说明：https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
4. SenseVoice 模型说明：https://modelscope.cn/models/iic/SenseVoiceSmall

## 十、总结

这个级联系统方案充分利用了 Paraformer 和 SenseVoice 各自的优势：
- ✅ Paraformer：擅长说话人区分和时间戳定位
- ✅ SenseVoice：擅长文本识别和情感标签

通过"先定位定人，再识别内容"的策略，实现了：
- ✅ 说话人区分功能
- ✅ 高准确率的文本识别
- ✅ 情感标签保留
- ✅ 不输出时间戳（根据用户需求）

该方案完全可行，建议按照上述计划逐步实施。


