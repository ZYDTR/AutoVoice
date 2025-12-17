"""
级联系统 v3：锚点分段 + 智能对齐
实现"用 Paraformer 定位定人，用 SenseVoice 修正内容"的方案

改进内容：
- 锚点分段：避免对齐漂移跨段传播
- 距离约束：防止幻觉导致的指针跳跃
- 幻觉检测：识别重复字符等异常输出
- 分层策略：单片段/多说话人/同说话人分别处理
"""

import os
import time
import re
import tempfile
import traceback
import difflib
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
DEFAULT_OUTPUT_DIR = "/Users/zhengyidi/AutoVoice/recordings"

# v3 配置
MAX_SEGMENT_DURATION_MS = 5 * 60 * 1000  # 最大对齐段时长：5分钟
MIN_SILENCE_GAP_MS = 2000  # 锚点条件：静音超过2秒
MIN_SIMILARITY_THRESHOLD = 0.5  # 最低相似度阈值
# ===========================================


def remove_emoji(text):
    """移除文本中的 emoji，保留标点符号和基本字符（包括中文）"""
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
    """移除 SenseVoice 输出的标签，只保留纯文本"""
    if not text:
        return ""
    tag_pattern = re.compile(r'<\s*\|[^|]*\|\s*>')
    text = tag_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_text(text):
    """
    文本标准化，用于模糊匹配
    - 移除标点符号
    - 转换为小写（针对英文）
    """
    if not text:
        return ""
    # 移除所有标点符号和空格
    text = re.sub(r'[，。！？、：；""''（）【】《》…—\s,.!?;:\'"()\[\]{}\n\r\t]+', '', text)
    # 转换为小写
    text = text.lower()
    return text


def extract_audio_segment(audio_path, start_ms, end_ms, buffer_ms=100):
    """提取音频片段"""
    start_ms = max(0, start_ms - buffer_ms)
    end_ms = end_ms + buffer_ms
    
    if USE_SOUNDFILE:
        try:
            audio_data, sample_rate = sf.read(audio_path)
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            segment = audio_data[start_sample:end_sample]
            return segment, sample_rate
        except Exception:
            pass
    
    if USE_LIBROSA:
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        duration = end_sec - start_sec
        audio_data, sample_rate = librosa.load(
            audio_path, offset=start_sec, duration=duration, sr=None
        )
        return audio_data, sample_rate
    
    raise RuntimeError("无法读取音频文件：soundfile 和 librosa 都不可用")


# ==================== v3 新增函数 ====================

def find_alignment_anchors(sentence_info, max_segment_duration_ms=MAX_SEGMENT_DURATION_MS):
    """
    识别对齐锚点
    
    锚点类型：
    1. 长静音段 (gap > MIN_SILENCE_GAP_MS)
    2. 说话人切换点
    3. 强制锚点 (每 max_segment_duration_ms 毫秒)
    
    返回：锚点索引列表（用于切分 sentence_info）
    """
    if not sentence_info:
        return [0, 0]
    
    anchors = [0]  # 起始锚点
    last_anchor_time = sentence_info[0]['start']
    
    for i in range(1, len(sentence_info)):
        prev = sentence_info[i-1]
        curr = sentence_info[i]
        
        # 锚点条件 1: 长静音
        gap = curr['start'] - prev['end']
        if gap > MIN_SILENCE_GAP_MS:
            anchors.append(i)
            last_anchor_time = curr['start']
            continue
        
        # 锚点条件 2: 说话人切换
        if prev.get('spk') != curr.get('spk'):
            anchors.append(i)
            last_anchor_time = curr['start']
            continue
        
        # 锚点条件 3: 强制间隔
        if curr['start'] - last_anchor_time > max_segment_duration_ms:
            anchors.append(i)
            last_anchor_time = curr['start']
    
    anchors.append(len(sentence_info))  # 结束锚点
    
    # 去重并排序
    anchors = sorted(list(set(anchors)))
    return anchors


def is_likely_hallucination(para_text, match_result=None, remaining_text_len=0):
    """
    检测 Paraformer 输出是否可能是幻觉
    
    幻觉特征：
    1. 文本很短但重复（如 "阿巴阿巴"）
    2. 在 SenseVoice 文本中找不到相似内容
    3. 匹配位置异常靠后
    """
    if not para_text:
        return True
    
    # 特征 1: 重复字符检测
    if len(para_text) >= 4:
        unique_chars = len(set(para_text))
        if unique_chars <= 2:  # 只有 1-2 种字符
            return True
    
    # 特征 2: 无匹配或低相似度
    if match_result is None:
        return True
    
    if match_result.get('similarity', 0) < 0.4:
        return True
    
    # 特征 3: 匹配位置异常（超过剩余文本的一半）
    if remaining_text_len > 0:
        if match_result.get('start_pos', 0) > remaining_text_len * 0.5:
            return True
    
    return False


def fuzzy_substring_search(haystack, needle, min_similarity=MIN_SIMILARITY_THRESHOLD, max_search_distance=None):
    """
    带距离约束的模糊子串搜索
    
    Args:
        haystack: 待搜索的文本（SenseVoice 输出）
        needle: 要查找的模式（Paraformer 片段文本）
        min_similarity: 最低相似度阈值
        max_search_distance: 最大搜索距离（字符数）
    
    Returns:
        匹配结果字典，或 None
    """
    needle_normalized = normalize_text(needle)
    
    if not needle_normalized:
        return None
    
    needle_len = len(needle_normalized)
    
    # 距离约束：默认为 needle 长度的 3 倍，最少 50 字符
    if max_search_distance is None:
        max_search_distance = max(needle_len * 3, 50)
    
    # 只搜索 haystack 的前 max_search_distance 个字符
    search_text = haystack[:max_search_distance]
    search_normalized = normalize_text(search_text)
    
    if not search_normalized:
        return None
    
    best_match = None
    best_score = 0
    
    # 滑动窗口搜索
    for window_size in range(
        max(1, int(needle_len * 0.5)),
        min(len(search_normalized), int(needle_len * 2)) + 1
    ):
        for start in range(len(search_normalized) - window_size + 1):
            candidate = search_normalized[start:start + window_size]
            score = difflib.SequenceMatcher(None, needle_normalized, candidate).ratio()
            
            if score > best_score:
                best_score = score
                best_match = {
                    'start_pos': start,
                    'end_pos': start + window_size,
                    'similarity': score
                }
    
    if best_match and best_match['similarity'] >= min_similarity:
        # 映射回原始文本（包含标点）
        original_start = map_to_original_pos(search_text, best_match['start_pos'])
        original_end = map_to_original_pos(search_text, best_match['end_pos'])
        
        return {
            'text': haystack[original_start:original_end],
            'start_pos': original_start,
            'end_pos': original_end,
            'similarity': best_match['similarity']
        }
    
    return None


def map_to_original_pos(original_text, normalized_pos):
    """将标准化文本中的位置映射回原始文本"""
    normalized_idx = 0
    for original_idx, char in enumerate(original_text):
        if not re.match(r'[，。！？、：；""''（）【】《》…—\s,.!?;:\'"()\[\]{}\n\r\t]', char):
            if normalized_idx == normalized_pos:
                return original_idx
            normalized_idx += 1
    return len(original_text)


def merge_same_speaker_segments(segments):
    """
    合并连续的同说话人片段
    减少需要对齐的单元数量，提高匹配准确度
    """
    if not segments:
        return []
    
    merged = []
    current_group = {
        'spk': segments[0].get('spk'),
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'text': segments[0].get('text', ''),
        'original_segments': [segments[0]]
    }
    
    for seg in segments[1:]:
        if seg.get('spk') == current_group['spk']:
            # 同说话人，合并
            current_group['end'] = seg['end']
            current_group['text'] += seg.get('text', '')
            current_group['original_segments'].append(seg)
        else:
            # 不同说话人，保存当前组，开始新组
            merged.append(current_group)
            current_group = {
                'spk': seg.get('spk'),
                'start': seg['start'],
                'end': seg['end'],
                'text': seg.get('text', ''),
                'original_segments': [seg]
            }
    
    merged.append(current_group)
    return merged


def group_by_speaker(segments):
    """按说话人分组（保持顺序）"""
    return merge_same_speaker_segments(segments)


def sequential_fuzzy_match(sensevoice_text, speaker_groups, log=print):
    """
    带顺序约束和距离约束的模糊匹配
    
    双重保护：
    1. 顺序约束：只在 current_pos 之后搜索
    2. 距离约束：只在合理范围内搜索，避免幻觉导致的跳跃
    """
    results = []
    current_pos = 0
    total_len = len(sensevoice_text)
    
    for group in speaker_groups:
        para_text = group.get('text', '')
        para_len = len(normalize_text(para_text))
        
        if para_len == 0:
            # 空文本，跳过
            results.append({
                'spk': group['spk'],
                'start': group['start'],
                'end': group['end'],
                'text': '',
                'original_segments': group.get('original_segments', []),
                'source': 'empty'
            })
            continue
        
        # 只在 current_pos 之后搜索
        remaining_text = sensevoice_text[current_pos:]
        remaining_len = len(remaining_text)
        
        # 计算合理的搜索距离
        max_search_distance = max(para_len * 3, 50)
        
        match = fuzzy_substring_search(
            haystack=remaining_text,
            needle=para_text,
            min_similarity=MIN_SIMILARITY_THRESHOLD,
            max_search_distance=max_search_distance
        )
        
        # 检查是否可能是幻觉
        if is_likely_hallucination(para_text, match, remaining_len):
            # 可能是幻觉，使用 Paraformer 原文，小步前进
            results.append({
                'spk': group['spk'],
                'start': group['start'],
                'end': group['end'],
                'text': remove_emoji(para_text),
                'original_segments': group.get('original_segments', []),
                'source': 'paraformer_hallucination'
            })
            # 小步前进
            current_pos += min(para_len, 20)
            continue
        
        if match:
            # 检查匹配位置是否合理
            if match['start_pos'] > max_search_distance * 0.8:
                # 匹配位置接近搜索边界，可能是误匹配
                results.append({
                    'spk': group['spk'],
                    'start': group['start'],
                    'end': group['end'],
                    'text': remove_emoji(para_text),
                    'original_segments': group.get('original_segments', []),
                    'source': 'paraformer_suspicious'
                })
                current_pos += min(para_len, 20)
            else:
                # 正常匹配
                absolute_end = current_pos + match['end_pos']
                matched_text = match['text'].strip()
                
                results.append({
                    'spk': group['spk'],
                    'start': group['start'],
                    'end': group['end'],
                    'text': matched_text if matched_text else remove_emoji(para_text),
                    'original_segments': group.get('original_segments', []),
                    'similarity': match['similarity'],
                    'source': 'sensevoice_fuzzy'
                })
                current_pos = absolute_end
        else:
            # 匹配失败，使用 Paraformer 原文
            results.append({
                'spk': group['spk'],
                'start': group['start'],
                'end': group['end'],
                'text': remove_emoji(para_text),
                'original_segments': group.get('original_segments', []),
                'source': 'paraformer_fallback'
            })
            current_pos += min(para_len, 20)
    
    return results


def expand_merged_results(merged_results):
    """
    将合并的结果展开回原始片段
    对于同说话人多片段，保持合并状态
    """
    expanded = []
    for r in merged_results:
        original_segments = r.get('original_segments', [])
        if len(original_segments) <= 1:
            # 单片段或无原始信息，直接添加
            expanded.append({
                'spk_id': r['spk'],
                'start': r['start'],
                'end': r['end'],
                'text': r['text'],
                'source': r.get('source', 'unknown')
            })
        else:
            # 多片段合并，保持合并状态输出
            expanded.append({
                'spk_id': r['spk'],
                'start': r['start'],
                'end': r['end'],
                'text': r['text'],
                'source': r.get('source', 'unknown'),
                'merged_count': len(original_segments)
            })
    return expanded


# ==================== 模型设置 ====================

def setup_cascaded_models():
    """初始化级联系统所需的模型"""
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


# ==================== v3 主处理函数 ====================

def process_audio_cascaded(audio_path, paraformer_model, sensevoice_model, log_callback=None, log_detail_callback=None):
    """
    级联处理 v3：锚点分段 + 智能对齐
    
    Args:
        audio_path: 音频文件路径
        paraformer_model: Paraformer + Cam++ 模型
        sensevoice_model: SenseVoice 模型
        log_callback: 日志回调函数（可选）
        log_detail_callback: 详细日志回调函数（可选）
    
    Returns:
        final_results: 最终结果列表
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
    
    total_start_time = time.time()
    
    # === 步骤 1: Paraformer 处理 ===
    log("="*60)
    log("🔄 步骤 1/4: 使用 Paraformer 进行说话人区分...")
    log("="*60)
    
    para_start_time = time.time()
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
    
    para_elapsed = time.time() - para_start_time
    log(f"✅ 检测到 {len(sentence_info)} 个句子片段，耗时: {para_elapsed:.2f}秒")
    
    # === 步骤 2: 识别对齐锚点 ===
    log("")
    log("="*60)
    log("🔄 步骤 2/4: 识别对齐锚点...")
    log("="*60)
    
    anchors = find_alignment_anchors(sentence_info)
    num_segments = len(anchors) - 1
    log(f"📊 识别到 {num_segments} 个对齐段")
    
    # 显示锚点分布
    for i in range(num_segments):
        start_idx = anchors[i]
        end_idx = anchors[i + 1]
        seg_count = end_idx - start_idx
        if seg_count > 0:
            start_time_ms = sentence_info[start_idx]['start']
            end_time_ms = sentence_info[end_idx - 1]['end']
            duration_sec = (end_time_ms - start_time_ms) / 1000
            log(f"  对齐段 {i+1}: {seg_count} 个片段, {duration_sec:.1f}秒 ({start_time_ms}ms - {end_time_ms}ms)", "sub")
    
    # === 步骤 3: 分段处理 ===
    log("")
    log("="*60)
    log("🔄 步骤 3/4: 分段处理（Paraformer + SenseVoice 对齐）...")
    log("="*60)
    
    all_results = []
    sense_total_time = 0
    
    for seg_idx in range(num_segments):
        start_idx = anchors[seg_idx]
        end_idx = anchors[seg_idx + 1]
        segment_infos = sentence_info[start_idx:end_idx]
        
        if not segment_infos:
            continue
        
        # 该段的时间范围
        seg_start_ms = segment_infos[0]['start']
        seg_end_ms = segment_infos[-1]['end']
        seg_duration = (seg_end_ms - seg_start_ms) / 1000
        
        log(f"处理对齐段 {seg_idx+1}/{num_segments}: {seg_duration:.1f}秒, {len(segment_infos)} 个片段")
        
        # 提取该段音频
        try:
            audio_segment, sr = extract_audio_segment(audio_path, seg_start_ms, seg_end_ms)
        except Exception as e:
            log(f"  ❌ 音频提取失败: {str(e)}", "error")
            # 降级：使用 Paraformer 原文
            for seg in segment_infos:
                all_results.append({
                    'spk_id': seg.get('spk', 'unknown'),
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': remove_emoji(seg.get('text', '')),
                    'source': 'paraformer_extract_failed'
                })
            continue
        
        # SenseVoice 处理该段
        sv_text = ""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, audio_segment, sr)
            
            sense_start = time.time()
            sv_res = sensevoice_model.generate(
                input=tmp_path,
                cache={},
                language="auto",
                use_itn=True,
            )
            sense_total_time += time.time() - sense_start
            
            os.unlink(tmp_path)
            
            # 提取 SenseVoice 文本
            if sv_res:
                if isinstance(sv_res, list) and len(sv_res) > 0:
                    item = sv_res[0]
                    if isinstance(item, dict):
                        sv_text = item.get('text', '')
                    elif isinstance(item, (list, tuple)) and len(item) > 0:
                        sv_text = item[0].get('text', '') if isinstance(item[0], dict) else str(item[0])
                    else:
                        sv_text = str(item)
                elif isinstance(sv_res, dict):
                    sv_text = sv_res.get('text', '')
            
            # 后处理
            if sv_text:
                sv_text = remove_sensevoice_tags(sv_text)
                sv_text = rich_transcription_postprocess(sv_text)
                sv_text = remove_emoji(sv_text)
                
        except Exception as e:
            log(f"  ❌ SenseVoice 处理失败: {str(e)}", "error")
            log_detail(traceback.format_exc(), "error")
        
        # === 步骤 4: 根据情况选择对齐策略 ===
        
        if not sv_text or not sv_text.strip():
            # SenseVoice 失败，使用 Paraformer 原文
            log(f"  ⚠️ SenseVoice 识别为空，使用 Paraformer 原文", "warning")
            for seg in segment_infos:
                text = remove_emoji(seg.get('text', ''))
                if text.strip():
                    all_results.append({
                        'spk_id': seg.get('spk', 'unknown'),
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': text,
                        'source': 'paraformer_sv_empty'
                    })
            continue
        
        if len(segment_infos) == 1:
            # 情况 A: 单片段 - 直接使用 SenseVoice 文本
            seg = segment_infos[0]
            log(f"  ✅ 单片段，直接使用 SenseVoice: {sv_text[:30]}..." if len(sv_text) > 30 else f"  ✅ 单片段: {sv_text}", "sub")
            all_results.append({
                'spk_id': seg.get('spk', 'unknown'),
                'start': seg['start'],
                'end': seg['end'],
                'text': sv_text,
                'source': 'sensevoice_direct'
            })
        
        else:
            # 检查是否多说话人
            speakers = set(s.get('spk') for s in segment_infos)
            
            if len(speakers) > 1:
                # 情况 B: 多说话人 - 按说话人分组后模糊匹配
                log(f"  🔀 多说话人 ({len(speakers)}人)，执行模糊匹配", "sub")
                speaker_groups = group_by_speaker(segment_infos)
                aligned = sequential_fuzzy_match(sv_text, speaker_groups, log)
                expanded = expand_merged_results(aligned)
                all_results.extend(expanded)
            
            else:
                # 情况 C: 同说话人多片段 - 保持合并输出
                log(f"  📝 同说话人 {len(segment_infos)} 个片段，保持合并", "sub")
                all_results.append({
                    'spk_id': segment_infos[0].get('spk', 'unknown'),
                    'start': segment_infos[0]['start'],
                    'end': segment_infos[-1]['end'],
                    'text': sv_text,
                    'source': 'sensevoice_merged',
                    'merged_count': len(segment_infos)
                })
    
    # === 完成 ===
    total_elapsed = time.time() - total_start_time
    log("")
    log("="*60)
    log(f"✅ 级联处理完成，总耗时: {total_elapsed:.2f}秒")
    log(f"   - Paraformer: {para_elapsed:.2f}秒")
    log(f"   - SenseVoice: {sense_total_time:.2f}秒")
    log("="*60)
    
    # 统计来源
    source_stats = {}
    for r in all_results:
        src = r.get('source', 'unknown')
        source_stats[src] = source_stats.get(src, 0) + 1
    
    log("📊 文本来源统计:")
    for src, count in sorted(source_stats.items()):
        log(f"   - {src}: {count}")
    
    return all_results


def format_cascaded_result(final_results, audio_file):
    """格式化级联系统的输出结果"""
    output_lines = []
    output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
    output_lines.append("="*60 + "\n")
    output_lines.append("📢 说话人区分结果（v3 锚点分段 + 智能对齐）:\n")
    output_lines.append("-"*60 + "\n")
    
    valid_results = [r for r in final_results if r.get('text', '').strip()]
    
    if not valid_results:
        output_lines.append("⚠️ 未检测到有效文本内容\n")
    else:
        for result in valid_results:
            spk_id = result['spk_id']
            text = result['text'].strip()
            source = result.get('source', '')
            merged = result.get('merged_count', 0)
            
            if text:
                line = f"说话人 {spk_id}: {text}"
                if merged > 1:
                    line += f" [合并{merged}段]"
                output_lines.append(line + "\n")
    
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
                    final_results = process_audio_cascaded(
                        audio_path, paraformer_model, sensevoice_model
                    )
                    
                    formatted_result = format_cascaded_result(final_results, audio_file)
                    print("\n" + formatted_result)
                    
                    output_file = os.path.join(
                        recordings_dir, 
                        f"{os.path.splitext(audio_file)[0]}_v3_transcription.txt"
                    )
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_result)
                    print(f"💾 结果已保存到: {output_file}\n")
                    
                except Exception as e:
                    print(f"❌ 处理文件 {audio_file} 时出错: {str(e)}")
                    traceback.print_exc()
            
            print(f"\n✅ 所有文件处理完成！")
