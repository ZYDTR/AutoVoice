# 改进的级联策略 v3：锚点分段 + 智能对齐

## 1. 已识别的问题

### 1.1 对齐漂移效应 ✅ 已解决
→ 锚点分段，漂移不跨段传播

### 1.2 重复词陷阱 ✅ 已解决  
→ 短段内匹配，上下文更明确

### 1.3 SenseVoice 长文本限制 ✅ 已解决
→ 分段处理，每段 2-8 分钟

### 1.4 线性比例假设 ❌ 新发现的问题

**问题描述**：

按时间比例分配文本假设语速恒定，这是错误的。

```
片段 A (2秒): "唉————"     → 1个字，但占2秒
片段 B (2秒): "吃葡萄不吐葡萄皮" → 8个字，也占2秒

SenseVoice: "唉吃葡萄不吐葡萄皮" (共9个字)

错误的时间比例分配：
├─ 片段 A (50%时间) → "唉吃葡萄" (4字) ❌ 
└─ 片段 B (50%时间) → "不吐葡萄皮" (5字) ❌

结果：字幕与声音完全错位
```

---

## 2. 新方案 v3：分层处理策略

### 2.1 核心原则

```
不是所有片段都需要精确对齐
根据场景选择最合适的策略
```

### 2.2 分层策略

```
┌────────────────────────────────────────────────────────────┐
│                    对齐段 (2-8分钟)                         │
└────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ 情况 A   │    │ 情况 B   │    │ 情况 C   │
       │ 单片段   │    │ 多说话人 │    │ 同说话人 │
       │          │    │          │    │ 多片段   │
       └──────────┘    └──────────┘    └──────────┘
              │               │               │
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ 直接使用 │    │ 按说话人 │    │ 保持合并 │
       │ 整段文本 │    │ 边界分割 │    │ 或模糊匹配│
       └──────────┘    └──────────┘    └──────────┘
```

---

## 3. 各情况的处理策略

### 3.1 情况 A：单片段段

最简单的情况，直接使用整个 SenseVoice 文本。

```python
if len(segment_infos) == 1:
    return [{
        'spk': segment_infos[0]['spk'],
        'start': segment_infos[0]['start'],
        'end': segment_infos[0]['end'],
        'text': sensevoice_text  # 直接使用
    }]
```

### 3.2 情况 B：多说话人段

说话人切换点是可靠的边界。按说话人分组后，使用**模糊子串匹配**定位每个说话人的文本。

```python
def align_multi_speaker_segment(speaker_groups, sensevoice_text):
    """
    多说话人段的对齐
    
    策略：用 Paraformer 的文本作为"查询"，在 SenseVoice 文本中模糊匹配
    """
    results = []
    sv_text = sensevoice_text
    
    for group in speaker_groups:
        para_text = group['text']  # Paraformer 的文本（作为查询）
        
        # 在 SenseVoice 文本中找最相似的子串
        match = fuzzy_substring_search(
            haystack=sv_text,
            needle=para_text,
            min_similarity=0.5
        )
        
        if match:
            matched_text = match['text']
            match_end = match['end_pos']
            
            results.append({
                'spk': group['spk'],
                'start': group['start'],
                'end': group['end'],
                'text': matched_text,
                'source': 'sensevoice'
            })
            
            # 从剩余文本中移除已匹配部分（避免重复匹配）
            sv_text = sv_text[match_end:]
        else:
            # 匹配失败，使用 Paraformer 原文
            results.append({
                'spk': group['spk'],
                'start': group['start'],
                'end': group['end'],
                'text': para_text,
                'source': 'paraformer_fallback'
            })
    
    return results
```

### 3.3 情况 C：同说话人多片段 ⭐ 关键改进

这是最棘手的情况。有两个选项：

#### 选项 C1：保持合并状态（推荐）

```python
def handle_same_speaker_multi_segment(segments, sensevoice_text):
    """
    同说话人多片段：保持合并，不做细粒度分配
    
    理由：
    - 细粒度分配必然引入假设（时间比例或文本比例）
    - 错误的分配比不分配更糟糕
    - 用户看到的是连续的同一说话人内容，合并输出是合理的
    """
    return [{
        'spk': segments[0]['spk'],
        'start': segments[0]['start'],
        'end': segments[-1]['end'],
        'text': sensevoice_text,
        'merged_count': len(segments),  # 标记合并了多少个原始片段
        'source': 'sensevoice_merged'
    }]
```

**优点**：
- 不会产生字幕错位
- SenseVoice 文本完整保留
- 用户体验：看到一整段同一人说的话

**缺点**：
- 失去 Paraformer 的细粒度时间戳
- 输出行数减少

#### 选项 C2：基于 Paraformer 文本长度比例

如果必须细分，用 Paraformer 文本长度比例（而非时间比例）：

```python
def handle_same_speaker_by_text_ratio(segments, sensevoice_text):
    """
    用 Paraformer 文本长度比例分配（而非时间比例）
    
    理由：
    - Paraformer 的文本长度分布反映了实际的词语分布
    - 比时间比例更准确
    """
    para_texts = [s.get('text', '') for s in segments]
    para_lengths = [len(t) for t in para_texts]
    total_para_len = sum(para_lengths)
    
    if total_para_len == 0:
        # 全空，使用选项 C1
        return handle_same_speaker_multi_segment(segments, sensevoice_text)
    
    sv_text = sensevoice_text
    sv_len = len(sv_text)
    results = []
    current_pos = 0
    
    for i, seg in enumerate(segments):
        para_len = para_lengths[i]
        ratio = para_len / total_para_len
        
        if i == len(segments) - 1:
            # 最后一个：取剩余所有
            allocated = sv_text[current_pos:]
        else:
            # 按比例分配
            char_count = int(sv_len * ratio)
            end_pos = find_word_boundary(sv_text, current_pos + char_count)
            allocated = sv_text[current_pos:end_pos]
            current_pos = end_pos
        
        results.append({
            'spk': seg['spk'],
            'start': seg['start'],
            'end': seg['end'],
            'text': allocated.strip(),
            'source': 'sensevoice_text_ratio'
        })
    
    return results
```

**对比**：

| 场景 | 时间比例 (错误) | 文本比例 (改进) |
|------|----------------|-----------------|
| "唉" (2秒) + "吃葡萄不吐葡萄皮" (2秒) | 各50%→错位 | 1:8→更准确 |
| "嗯" (0.5秒) + "好的我知道了" (2秒) | 20%:80%→还行 | 1:5→更准确 |

---

## 4. 模糊子串匹配算法

### 4.1 核心思想

用 Paraformer 文本作为"锚点"，在 SenseVoice 文本中找最相似位置。

```python
def fuzzy_substring_search(haystack, needle, min_similarity=0.5):
    """
    在 haystack 中找与 needle 最相似的子串
    
    参数：
        haystack: SenseVoice 的完整文本
        needle: Paraformer 的某个片段文本
        min_similarity: 最低相似度阈值
    
    返回：
        {text: 匹配的子串, start_pos, end_pos, similarity}
        或 None（未找到）
    """
    import difflib
    
    needle_normalized = normalize_text(needle)
    haystack_normalized = normalize_text(haystack)
    
    if not needle_normalized:
        return None
    
    needle_len = len(needle_normalized)
    best_match = None
    best_score = 0
    
    # 滑动窗口搜索
    # 搜索范围：needle 长度的 0.5x 到 2x
    for window_size in range(
        max(1, int(needle_len * 0.5)),
        min(len(haystack_normalized), int(needle_len * 2)) + 1
    ):
        for start in range(len(haystack_normalized) - window_size + 1):
            candidate = haystack_normalized[start:start + window_size]
            
            # 计算相似度
            score = difflib.SequenceMatcher(
                None, needle_normalized, candidate
            ).ratio()
            
            if score > best_score:
                best_score = score
                best_match = {
                    'start_pos': start,
                    'end_pos': start + window_size,
                    'similarity': score
                }
    
    if best_match and best_match['similarity'] >= min_similarity:
        # 映射回原始文本（包含标点）
        original_start = map_to_original_pos(haystack, best_match['start_pos'])
        original_end = map_to_original_pos(haystack, best_match['end_pos'])
        
        return {
            'text': haystack[original_start:original_end],
            'start_pos': original_start,
            'end_pos': original_end,
            'similarity': best_match['similarity']
        }
    
    return None
```

### 4.2 顺序约束优化

为了避免匹配跳跃（比如匹配到后面的重复词），添加顺序约束：

```python
def sequential_fuzzy_match(sensevoice_text, paraformer_segments):
    """
    带顺序约束的模糊匹配
    
    关键：每次匹配后，只在剩余文本中搜索下一个
    这避免了匹配"跳跃"到后面的重复词
    """
    results = []
    current_pos = 0
    
    for seg in paraformer_segments:
        # 只在 current_pos 之后搜索
        remaining_text = sensevoice_text[current_pos:]
        
        match = fuzzy_substring_search(
            haystack=remaining_text,
            needle=seg['text'],
            min_similarity=0.5
        )
        
        if match:
            # 更新位置指针
            absolute_start = current_pos + match['start_pos']
            absolute_end = current_pos + match['end_pos']
            
            results.append({
                'spk': seg['spk'],
                'start': seg['start'],
                'end': seg['end'],
                'text': match['text'],
                'similarity': match['similarity'],
                'source': 'sensevoice_fuzzy'
            })
            
            # 移动指针到匹配结束位置
            current_pos = absolute_end
        else:
            # 匹配失败，使用 Paraformer 原文
            results.append({
                'spk': seg['spk'],
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'source': 'paraformer_fallback'
            })
            
            # 即使失败，也要估算移动位置（基于文本长度）
            estimated_move = len(seg.get('text', '')) * 1.2
            current_pos += int(estimated_move)
    
    return results
```

---

## 5. 完整处理流程 v3

```python
def process_audio_cascaded_v3(audio_path, paraformer_model, sensevoice_model, log=print):
    """
    级联处理 v3：锚点分段 + 智能对齐
    """
    
    # === 步骤 1: Paraformer 处理 ===
    sentence_info = run_paraformer(audio_path, paraformer_model)
    
    # === 步骤 2: 识别对齐锚点 ===
    anchors = find_alignment_anchors(sentence_info)
    
    # === 步骤 3: 按锚点切分并处理 ===
    all_results = []
    
    for seg_idx in range(len(anchors) - 1):
        segment_infos = sentence_info[anchors[seg_idx]:anchors[seg_idx + 1]]
        
        # 提取该段音频
        audio_segment = extract_segment_audio(...)
        
        # SenseVoice 处理
        sv_text = run_sensevoice_on_segment(audio_segment, sensevoice_model)
        
        if not sv_text.strip():
            # SenseVoice 失败，使用 Paraformer 原文
            for seg in segment_infos:
                all_results.append({...seg, 'source': 'paraformer'})
            continue
        
        # === 步骤 4: 根据情况选择对齐策略 ===
        
        if len(segment_infos) == 1:
            # 情况 A: 单片段
            all_results.append({
                'spk': segment_infos[0]['spk'],
                'start': segment_infos[0]['start'],
                'end': segment_infos[0]['end'],
                'text': sv_text,
                'source': 'sensevoice_direct'
            })
        
        else:
            # 检查是否多说话人
            speakers = set(s.get('spk') for s in segment_infos)
            
            if len(speakers) > 1:
                # 情况 B: 多说话人
                # 按说话人分组
                speaker_groups = group_by_speaker(segment_infos)
                aligned = sequential_fuzzy_match(sv_text, speaker_groups)
                all_results.extend(aligned)
            
            else:
                # 情况 C: 同说话人多片段
                # 选择策略：
                
                # 选项 C1: 保持合并（推荐）
                all_results.append({
                    'spk': segment_infos[0]['spk'],
                    'start': segment_infos[0]['start'],
                    'end': segment_infos[-1]['end'],
                    'text': sv_text,
                    'merged_count': len(segment_infos),
                    'source': 'sensevoice_merged'
                })
                
                # 或选项 C2: 基于文本比例分配
                # aligned = handle_same_speaker_by_text_ratio(segment_infos, sv_text)
                # all_results.extend(aligned)
    
    return {'segments': all_results}
```

---

## 6. 输出格式说明

### 6.1 带来源标记

每个输出片段标记其文本来源：

| source | 含义 |
|--------|------|
| `sensevoice_direct` | 单片段，直接使用 SenseVoice |
| `sensevoice_fuzzy` | 多说话人，模糊匹配成功 |
| `sensevoice_merged` | 同说话人多片段，保持合并 |
| `sensevoice_text_ratio` | 同说话人多片段，按文本比例分配 |
| `paraformer_fallback` | SenseVoice 匹配失败，使用 Paraformer 原文 |

### 6.2 合并片段的展开（可选）

如果用户需要细粒度输出，可以后处理展开合并片段：

```python
def expand_merged_segments(results, original_sentence_info):
    """
    将合并的片段按原始时间戳展开
    文本均匀分配（或保持空）
    """
    expanded = []
    for r in results:
        if r.get('merged_count', 1) == 1:
            expanded.append(r)
        else:
            # 找到对应的原始片段
            originals = find_originals_in_range(
                original_sentence_info,
                r['start'], r['end']
            )
            # 均匀分配文本（或只在第一个片段放文本）
            for i, orig in enumerate(originals):
                expanded.append({
                    'spk': orig['spk'],
                    'start': orig['start'],
                    'end': orig['end'],
                    'text': r['text'] if i == 0 else '',  # 或均匀分配
                    'source': r['source']
                })
    return expanded
```

---

## 7. 方案对比总结

| 问题 | v1 原方案 | v2 锚点分段 | v3 智能对齐 |
|------|-----------|-------------|-------------|
| 对齐漂移 | ❌ 传播 | ✅ 不跨段 | ✅ 不跨段 |
| 重复词陷阱 | ❌ 难区分 | ✅ 短段内 | ✅ 顺序约束 |
| 长文本限制 | ❌ 整体处理 | ✅ 分段 | ✅ 分段 |
| 线性比例假设 | ❌ 时间比例 | ❌ 时间比例 | ✅ 文本比例或合并 |
| 细粒度准确性 | ❌ | ❌ | ✅ 模糊匹配 |

---

## 8. 推荐的默认策略

```
对于大多数场景，推荐：

1. 单片段段 → 直接使用 SenseVoice 文本
2. 多说话人段 → 顺序模糊匹配
3. 同说话人多片段 → 保持合并输出

理由：
- 保持合并避免了所有分配假设带来的风险
- 用户看到的是一整段同一人说的话，符合直觉
- 如需细粒度，可后处理展开
```

---

**确认后开始实施代码修改。**
