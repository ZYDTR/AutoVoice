# GUI 与 process_single_cascaded.py 逻辑对比

## 概述
本文档对比 GUI 中级联模式的处理逻辑和 `process_single_cascaded.py` 脚本的处理逻辑。

## 1. 模型加载

### GUI (`run_m1_sensevoice_gui.py`)
```python
# 在 load_model_async() 中
paraformer_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    device=DEVICE,
    ncpu=THREADS,
    disable_update=True
)

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
```

### process_single_cascaded.py
```python
# 使用 setup_cascaded_models()
paraformer_model, sensevoice_model = setup_cascaded_models()
```

**结论**：✅ **一致** - `setup_cascaded_models()` 使用相同的参数加载模型

## 2. 音频处理

### GUI (`process_audio_cascaded()`)
```python
final_results = cascaded_process(
    audio_file, 
    self.paraformer_model, 
    self.sensevoice_model,
    log_callback=self.log,              # ✅ 传入日志回调
    log_detail_callback=self.log_detail # ✅ 传入详细日志回调
)
```

### process_single_cascaded.py
```python
final_results = process_audio_cascaded(
    audio_file, 
    paraformer_model, 
    sensevoice_model
    # ❌ 没有传入日志回调（日志会直接 print）
)
```

**结论**：⚠️ **有差异** - GUI 传入日志回调，脚本没有（但处理逻辑相同）

## 3. 结果格式化

### GUI (`format_result_with_speaker()`)
```python
# 检测级联模式结果
if isinstance(speaker_info, list) and len(speaker_info) > 0:
    first_item = speaker_info[0]
    if isinstance(first_item, dict) and "spk_id" in first_item:
        is_cascaded_result = True

if is_cascaded_result:
    for item in speaker_info:
        spk_id = item.get("spk_id", "Unknown")
        text = item.get("text", "")
        text = self.remove_emoji(text)  # ⚠️ 再次移除 emoji（冗余）
        output_lines.append(f"说话人 {spk_id}: {text}\n")
```

### process_single_cascaded.py (`format_cascaded_result()`)
```python
# 过滤掉空文本的结果
valid_results = [r for r in final_results if r.get('text', '').strip()]

for result in valid_results:
    spk_id = result['spk_id']
    text = result['text'].strip()
    # ✅ 文本已经在 process_audio_cascaded() 中移除了 emoji
    if text:
        output_lines.append(f"说话人 {spk_id}: {text}\n")
```

**结论**：✅ **基本一致** - GUI 中会再次移除 emoji（虽然冗余，但不影响结果）

## 4. 文件保存

### GUI
```python
output_file = os.path.join(self.output_dir, f"{base_name}_transcription.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(formatted_result)
```

### process_single_cascaded.py
```python
output_file = os.path.join(
    os.path.dirname(audio_file),
    f"{os.path.splitext(os.path.basename(audio_file))[0]}_cascaded_transcription.txt"
)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(formatted_result)
```

**结论**：⚠️ **有差异** - 文件名不同（GUI 用 `_transcription.txt`，脚本用 `_cascaded_transcription.txt`）

## 总结

### ✅ 一致的方面
1. **模型加载参数**：完全相同
2. **核心处理逻辑**：使用相同的 `process_audio_cascaded()` 函数
3. **结果格式化**：逻辑相同（GUI 中会冗余地移除 emoji，但不影响结果）

### ⚠️ 差异的方面
1. **日志输出**：
   - GUI：通过回调函数输出到界面
   - 脚本：直接 print 到终端
   - **影响**：无（只是输出方式不同）

2. **文件名**：
   - GUI：`{base_name}_transcription.txt`
   - 脚本：`{base_name}_cascaded_transcription.txt`
   - **影响**：文件名不同，但内容相同

3. **emoji 移除**：
   - GUI：在格式化时再次移除 emoji（冗余）
   - 脚本：只在处理时移除一次
   - **影响**：无（结果相同）

## 结论

**`process_single_cascaded.py` 和 GUI 的处理逻辑基本一致**，主要差异在于：
- 日志输出方式（GUI 显示在界面，脚本输出到终端）
- 输出文件名（GUI 用 `_transcription.txt`，脚本用 `_cascaded_transcription.txt`）

**核心处理逻辑完全相同**，所以两者的识别结果应该是一致的。

