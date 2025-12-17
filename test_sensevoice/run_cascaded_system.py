"""
级联系统：先 Paraformer 做 Diarization，再用 SenseVoice 识别文本
实现"用 Paraformer 定位定人，用 SenseVoice 修正内容"的方案
"""

import os
import time
import re
import tempfile
import traceback
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 尝试导入音频处理库（优先使用 soundfile，如果不支持格式则使用 librosa）
try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    USE_SOUNDFILE = False

try:
    import librosa
    USE_LIBROSA = True
except ImportError:
    USE_LIBROSA = False
    if not USE_SOUNDFILE:
        raise ImportError("需要安装 soundfile 或 librosa 库")

# ================= 配置区域 =================
DEVICE = "cpu"
THREADS = 4
DEFAULT_OUTPUT_DIR = "/Users/zhengyidi/AutoVoice/recordings"  # 默认输出目录
# ===========================================

def remove_emoji(text):
    """移除文本中的 emoji，保留标点符号和基本字符（包括中文）"""
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

def remove_sensevoice_tags(text):
    """
    移除 SenseVoice 输出的标签，只保留纯文本
    
    移除的标签格式：
    - <|en|>, <|zh|>, <|yue|>, <|ja|> 等语言标签
    - <|NEUTRAL|>, <|EMO_UNKNOWN|> 等情绪标签
    - <|Speech|>, <|within|> 等其他标签
    
    注意：标签格式可能是 <|...|> 或 < | ... | >（标签内可能有空格）
    """
    if not text:
        return ""
    
    # 移除所有 <|...|> 格式的标签（包括标签内可能有空格的情况）
    # 匹配 <|...|> 或 < | ... | > 等格式
    tag_pattern = re.compile(r'<\s*\|[^|]*\|\s*>')
    text = tag_pattern.sub('', text)
    
    # 清理多余的空格（多个连续空格变成一个）
    text = re.sub(r'\s+', ' ', text)
    
    # 清理首尾空格
    text = text.strip()
    
    return text

def extract_audio_segment(audio_path, start_ms, end_ms, buffer_ms=100):
    """
    提取音频片段
    
    Args:
        audio_path: 音频文件路径
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        buffer_ms: 前后缓冲时间（毫秒），默认100ms
    
    Returns:
        audio_segment: 音频数据（numpy array）
        sample_rate: 采样率
    """
    # 添加前后缓冲
    start_ms = max(0, start_ms - buffer_ms)
    end_ms = end_ms + buffer_ms
    
    # 尝试使用 soundfile（速度快，但格式支持有限）
    if USE_SOUNDFILE:
        try:
            audio_data, sample_rate = sf.read(audio_path)
            # 转换为采样点索引
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            # 确保不超出范围
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            # 提取片段
            segment = audio_data[start_sample:end_sample]
            return segment, sample_rate
        except Exception:
            # soundfile 不支持该格式，降级使用 librosa
            pass
    
    # 使用 librosa（支持更多格式，但可能较慢）
    if USE_LIBROSA:
        # librosa 使用秒作为单位
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        duration = end_sec - start_sec
        
        # 读取指定时间段的音频
        audio_data, sample_rate = librosa.load(
            audio_path,
            offset=start_sec,
            duration=duration,
            sr=None  # 保持原始采样率
        )
        
        return audio_data, sample_rate
    
    raise RuntimeError("无法读取音频文件：soundfile 和 librosa 都不可用")

def setup_cascaded_models():
    """
    初始化级联系统所需的模型
    """
    print("🔄 正在加载 Paraformer + Cam++ 模型...")
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
    print(f"✅ Paraformer + Cam++ 模型加载完成，耗时: {elapsed:.2f}秒")
    
    print("🔄 正在加载 SenseVoice 模型...")
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
    print(f"✅ SenseVoice 模型加载完成，耗时: {elapsed:.2f}秒")
    
    return paraformer_model, sensevoice_model

def process_audio_cascaded(audio_path, paraformer_model, sensevoice_model, log_callback=None, log_detail_callback=None):
    """
    级联处理音频：先 Paraformer 做 diarization，再用 SenseVoice 识别
    
    Args:
        audio_path: 音频文件路径
        paraformer_model: Paraformer + Cam++ 模型
        sensevoice_model: SenseVoice 模型
        log_callback: 日志回调函数（可选），用于 GUI 显示
        log_detail_callback: 详细日志回调函数（可选），用于 GUI 显示
    
    Returns:
        final_results: 最终结果列表，每个元素包含 spk_id, start, end, text
    """
    def log(msg, level="main"):
        if log_callback:
            log_callback(msg, level)
        else:
            print(f"[{level}] {msg}")
    
    def log_detail(msg, level="info"):
        if log_detail_callback:
            log_detail_callback(msg, level)
        else:
            print(f"[{level}] {msg}")
    
    # === 步骤 1: Paraformer 处理（获取时间戳和说话人ID） ===
    log("="*60)
    log("🔄 步骤 1/3: 使用 Paraformer 进行说话人区分...")
    log("="*60)
    
    start_time = time.time()
    paraformer_res = paraformer_model.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )
    
    if not paraformer_res or len(paraformer_res) == 0:
        raise ValueError("Paraformer 处理失败：未返回结果")
    
    paraformer_result = paraformer_res[0]
    sentence_info = paraformer_result.get('sentence_info', [])
    
    if not sentence_info:
        raise ValueError("Paraformer 处理失败：未检测到句子信息")
    
    elapsed = time.time() - start_time
    log(f"✅ 检测到 {len(sentence_info)} 个句子片段，耗时: {elapsed:.2f}秒")
    
    # === 步骤 2: 将片段按30秒窗口分组，合并后统一用 SenseVoice 识别 ===
    log("")
    log("="*60)
    log("🔄 步骤 2/3: 将片段按30秒窗口分组，合并后用 SenseVoice 批量识别...")
    log("="*60)
    
    # SenseVoice 内部以30秒为最优片段长度，所以我们按30秒窗口分组
    SEGMENT_WINDOW_SEC = 30  # 30秒窗口
    SEGMENT_WINDOW_MS = SEGMENT_WINDOW_SEC * 1000
    
    # 将 sentence_info 按30秒窗口分组
    window_groups = []
    current_window_start = 0
    current_group = []
    
    for sent_info in sentence_info:
        start_ms = sent_info['start']
        end_ms = sent_info['end']
        
        # 如果当前片段超出当前窗口，开始新窗口
        if start_ms >= current_window_start + SEGMENT_WINDOW_MS:
            if current_group:
                window_groups.append({
                    'window_start': current_window_start,
                    'window_end': current_window_start + SEGMENT_WINDOW_MS,
                    'segments': current_group
                })
            current_window_start = (start_ms // SEGMENT_WINDOW_MS) * SEGMENT_WINDOW_MS
            current_group = []
        
        current_group.append(sent_info)
    
    # 添加最后一组
    if current_group:
        window_groups.append({
            'window_start': current_window_start,
            'window_end': current_window_start + SEGMENT_WINDOW_MS,
            'segments': current_group
        })
    
    log(f"📊 将 {len(sentence_info)} 个片段分组为 {len(window_groups)} 个30秒窗口")
    for i, group in enumerate(window_groups, 1):
        log(f"  窗口 {i}: {len(group['segments'])} 个片段，时间范围: {group['window_start']}ms - {group['window_end']}ms", "sub")
    
    sensevoice_results = []
    total_start_time = time.time()
    
    # 提取并合并30秒窗口的音频
    log(f"准备提取并合并 {len(window_groups)} 个30秒窗口的音频...")
    extract_start_time = time.time()
    
    window_audio_files = []
    window_info_list = []
    
    for window_idx, window_group in enumerate(window_groups):
        window_start_ms = window_group['window_start']
        window_end_ms = window_group['window_end']
        segments = window_group['segments']
        
        try:
            # 提取整个30秒窗口的音频
            audio_segment, sample_rate = extract_audio_segment(
                audio_path, window_start_ms, window_end_ms
            )
            
            # 如果音频长度不足30秒，用静音填充（保持原始长度也可以）
            expected_samples = int(SEGMENT_WINDOW_SEC * sample_rate)
            if len(audio_segment) < expected_samples:
                # 用静音填充到30秒
                padding_samples = expected_samples - len(audio_segment)
                if len(audio_segment.shape) == 1:
                    # 单声道
                    padding = np.zeros(padding_samples, dtype=audio_segment.dtype)
                else:
                    # 多声道
                    padding = np.zeros((padding_samples, audio_segment.shape[1]), dtype=audio_segment.dtype)
                audio_segment = np.concatenate([audio_segment, padding])
            
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                if USE_SOUNDFILE:
                    sf.write(tmp_path, audio_segment, sample_rate)
                elif USE_LIBROSA:
                    import soundfile as sf_write
                    sf_write.write(tmp_path, audio_segment, sample_rate)
                else:
                    raise RuntimeError("无法写入音频文件：soundfile 不可用")
            
            window_audio_files.append(tmp_path)
            window_info_list.append({
                'window_idx': window_idx,
                'window_start': window_start_ms,
                'window_end': window_end_ms,
                'segments': segments  # 保存原始片段信息，用于后续映射
            })
            
        except Exception as e:
            log(f"  ❌ 提取窗口 {window_idx+1} 时出错: {str(e)}", "error")
            log_detail(f"提取窗口 {window_idx+1} 时出错: {str(e)}", "error")
            log_detail(traceback.format_exc(), "error")
            # 降级：使用 Paraformer 的文本
            for seg_info in segments:
                text = seg_info.get('text', '')
                text = remove_emoji(text)
                sensevoice_results.append({
                    'spk_id': seg_info.get('spk', 'unknown'),
                    'start': seg_info['start'],
                    'end': seg_info['end'],
                    'text': text
                })
    
    extract_time = time.time() - extract_start_time
    log(f"✅ 30秒窗口音频提取完成，耗时: {extract_time:.2f}秒")
    
    # 批量处理 SenseVoice 识别（每批处理多个30秒窗口）
    BATCH_SIZE = 8  # 每批处理8个30秒窗口
    log(f"开始批量识别（每批 {BATCH_SIZE} 个30秒窗口）...")
    sense_start_time = time.time()
    
    for batch_start in range(0, len(window_audio_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(window_audio_files))
        batch_files = window_audio_files[batch_start:batch_end]
        batch_info = window_info_list[batch_start:batch_end]
        
        log(f"处理批次 {batch_start//BATCH_SIZE + 1}/{(len(window_audio_files) + BATCH_SIZE - 1)//BATCH_SIZE}: 窗口 {batch_start+1}-{batch_end}")
        
        try:
            # 批量调用 SenseVoice 处理30秒窗口
            batch_sense_res = sensevoice_model.generate(
                input=batch_files,
                cache={},
                language="auto",
                use_itn=True,
            )
            
            # 处理每个30秒窗口的结果，映射回原始小片段
            for i, (window_info, sense_res) in enumerate(zip(batch_info, batch_sense_res)):
                window_idx = window_info['window_idx']
                window_start_ms = window_info['window_start']
                window_end_ms = window_info['window_end']
                segments = window_info['segments']
                
                # 提取 SenseVoice 的文本
                window_text = ""
                if sense_res:
                    if isinstance(sense_res, list):
                        if len(sense_res) > 0:
                            result_item = sense_res[0]
                            if isinstance(result_item, dict):
                                window_text = result_item.get('text', '')
                            else:
                                window_text = str(result_item)
                    elif isinstance(sense_res, dict):
                        window_text = sense_res.get('text', '')
                    else:
                        window_text = str(sense_res) if sense_res else ""
                
                # 后处理：移除标签和 emoji
                if window_text and window_text.strip():
                    window_text = remove_sensevoice_tags(window_text)
                    window_text = rich_transcription_postprocess(window_text)
                    window_text = remove_emoji(window_text)
                
                # 将30秒窗口的文本映射回原始小片段
                # 策略：如果窗口内只有一个片段，直接使用整个文本
                # 如果有多个片段，按时间比例分配文本（简化处理）
                if window_text and window_text.strip():
                    if len(segments) == 1:
                        # 只有一个片段，直接使用整个文本
                        seg_info = segments[0]
                        text = window_text
                        spk_id = seg_info.get('spk', 'unknown')
                        start_ms = seg_info['start']
                        end_ms = seg_info['end']
                        
                        if text and text.strip():
                            log(f"  窗口 {window_idx+1} 片段 1 (说话人 {spk_id}): {text[:50]}..." if len(text) > 50 else f"  窗口 {window_idx+1} 片段 1 (说话人 {spk_id}): {text}", "sub")
                            sensevoice_results.append({
                                'spk_id': spk_id,
                                'start': start_ms,
                                'end': end_ms,
                                'text': text.strip()
                            })
                    else:
                        # 多个片段：使用 Paraformer 的文本（因为无法准确分割 SenseVoice 的文本）
                        log(f"  窗口 {window_idx+1} 包含 {len(segments)} 个片段，使用 Paraformer 文本", "sub")
                        for seg_info in segments:
                            text = seg_info.get('text', '')
                            text = remove_emoji(text)
                            if text and text.strip():
                                sensevoice_results.append({
                                    'spk_id': seg_info.get('spk', 'unknown'),
                                    'start': seg_info['start'],
                                    'end': seg_info['end'],
                                    'text': text.strip()
                                })
                else:
                    # SenseVoice 识别失败，降级使用 Paraformer 文本
                    log(f"  窗口 {window_idx+1} SenseVoice 识别失败，使用 Paraformer 文本", "warning")
                    for seg_info in segments:
                        text = seg_info.get('text', '')
                        text = remove_emoji(text)
                        if text and text.strip():
                            sensevoice_results.append({
                                'spk_id': seg_info.get('spk', 'unknown'),
                                'start': seg_info['start'],
                                'end': seg_info['end'],
                                'text': text.strip()
                            })
                
        except Exception as e:
            log(f"  ❌ 批次处理出错: {str(e)}", "error")
            log_detail(f"批次处理出错: {str(e)}", "error")
            log_detail(traceback.format_exc(), "error")
            # 降级处理：逐个处理这个批次的窗口
            for window_info in batch_info:
                window_idx = window_info['window_idx']
                segments = window_info['segments']
                tmp_path = window_audio_files[batch_start + batch_info.index(window_info)]
                
                try:
                    single_res = sensevoice_model.generate(
                        input=tmp_path,
                        cache={},
                        language="auto",
                        use_itn=True,
                    )
                    
                    window_text = ""
                    if single_res:
                        if isinstance(single_res, list):
                            if len(single_res) > 0:
                                result_item = single_res[0]
                                if isinstance(result_item, dict):
                                    window_text = result_item.get('text', '')
                                else:
                                    window_text = str(result_item)
                        elif isinstance(single_res, dict):
                            window_text = single_res.get('text', '')
                        else:
                            window_text = str(single_res) if single_res else ""
                    
                    if window_text and window_text.strip():
                        window_text = remove_sensevoice_tags(window_text)
                        window_text = rich_transcription_postprocess(window_text)
                        window_text = remove_emoji(window_text)
                        
                        # 如果只有一个片段，直接使用
                        if len(segments) == 1 and window_text:
                            seg_info = segments[0]
                            sensevoice_results.append({
                                'spk_id': seg_info.get('spk', 'unknown'),
                                'start': seg_info['start'],
                                'end': seg_info['end'],
                                'text': window_text.strip()
                            })
                        else:
                            # 多个片段，使用 Paraformer 文本
                            for seg_info in segments:
                                text = seg_info.get('text', '')
                                text = remove_emoji(text)
                                if text and text.strip():
                                    sensevoice_results.append({
                                        'spk_id': seg_info.get('spk', 'unknown'),
                                        'start': seg_info['start'],
                                        'end': seg_info['end'],
                                        'text': text.strip()
                                    })
                    else:
                        # 使用 Paraformer 文本
                        for seg_info in segments:
                            text = seg_info.get('text', '')
                            text = remove_emoji(text)
                            if text and text.strip():
                                sensevoice_results.append({
                                    'spk_id': seg_info.get('spk', 'unknown'),
                                    'start': seg_info['start'],
                                    'end': seg_info['end'],
                                    'text': text.strip()
                                })
                except Exception as e2:
                    # 最终降级：使用 Paraformer 文本
                    for seg_info in segments:
                        text = seg_info.get('text', '')
                        text = remove_emoji(text)
                        if text and text.strip():
                            sensevoice_results.append({
                                'spk_id': seg_info.get('spk', 'unknown'),
                                'start': seg_info['start'],
                                'end': seg_info['end'],
                                'text': text.strip()
                            })
        
        finally:
            # 清理这个批次的临时文件
            for tmp_path in batch_files:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
    
    sense_time = time.time() - sense_start_time
    log(f"✅ SenseVoice 批量识别完成，耗时: {sense_time:.2f}秒")
    
    total_elapsed = time.time() - total_start_time
    log("")
    log(f"✅ 所有片段处理完成，总耗时: {total_elapsed:.2f}秒")
    
    return sensevoice_results

def format_cascaded_result(final_results, audio_file):
    """
    格式化级联系统的输出结果
    """
    output_lines = []
    output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
    output_lines.append("="*60 + "\n")
    output_lines.append("📢 说话人区分结果（使用 SenseVoice 识别）:\n")
    output_lines.append("-"*60 + "\n")
    
    # 过滤掉空文本的结果
    valid_results = [r for r in final_results if r.get('text', '').strip()]
    
    if not valid_results:
        output_lines.append("⚠️ 未检测到有效文本内容\n")
    else:
        for result in valid_results:
            spk_id = result['spk_id']
            text = result['text'].strip()
            
            # 只输出非空文本
            if text:
                output_lines.append(f"说话人 {spk_id}: {text}\n")
    
    output_lines.append("\n" + "="*60 + "\n")
    
    return "".join(output_lines)

if __name__ == "__main__":
    # 1. 加载模型
    paraformer_model, sensevoice_model = setup_cascaded_models()
    
    # 2. 处理 recordings 目录下的所有音频文件
    recordings_dir = DEFAULT_OUTPUT_DIR
    
    if not os.path.exists(recordings_dir):
        print(f"❌ 错误: 目录 {recordings_dir} 不存在")
    else:
        # 获取所有音频文件
        audio_files = [f for f in os.listdir(recordings_dir) 
                      if f.endswith(('.webm', '.mp3', '.wav', '.m4a', '.flac'))]
        
        if not audio_files:
            print(f"⚠️ 在 {recordings_dir} 中未找到音频文件")
        else:
            print(f"\n📁 找到 {len(audio_files)} 个音频文件，开始处理...\n")
            
            for idx, audio_file in enumerate(sorted(audio_files), 1):
                audio_path = os.path.join(recordings_dir, audio_file)
                print(f"\n{'='*60}")
                print(f"文件 {idx}/{len(audio_files)}: {audio_file}")
                print(f"{'='*60}")
                
                try:
                    # 级联处理
                    final_results = process_audio_cascaded(
                        audio_path, paraformer_model, sensevoice_model
                    )
                    
                    # 格式化输出
                    formatted_result = format_cascaded_result(final_results, audio_file)
                    
                    print("\n" + formatted_result)
                    
                    # 保存结果到文件
                    output_file = os.path.join(
                        recordings_dir, 
                        f"{os.path.splitext(audio_file)[0]}_cascaded_transcription.txt"
                    )
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_result)
                    print(f"💾 结果已保存到: {output_file}\n")
                    
                except Exception as e:
                    print(f"❌ 处理文件 {audio_file} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n✅ 所有文件处理完成！")

