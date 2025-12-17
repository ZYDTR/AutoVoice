#!/usr/bin/env python3
"""
单独使用 SenseVoice 处理指定音频文件的测试脚本
"""
import os
import time
import torch
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ================= 配置区域 =================
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
DEVICE = "cpu" 
THREADS = 4 

def remove_sensevoice_tags(text):
    """
    移除 SenseVoice 输出的标签，只保留纯文本
    """
    if not text:
        return ""
    
    # 移除所有 <|...|> 格式的标签（包括标签内可能有空格的情况）
    tag_pattern = re.compile(r'<\s*\|[^|]*\|\s*>')
    text = tag_pattern.sub('', text)
    
    # 清理多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_emoji(text):
    """移除文本中的 emoji，保留标点符号和基本字符（包括中文）"""
    # 使用更精确的 emoji 范围，避免误删中文字符
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
    # 移除 emoji，但保留所有其他字符（包括中文、日文等）
    return emoji_pattern.sub('', text).strip()

def setup_model():
    print(f"🔄 正在初始化 SenseVoice 模型 (Device: {DEVICE})...")
    
    start_time = time.time()
    
    # AutoModel 初始化
    model_kwargs = {
        "model": SENSEVOICE_MODEL,
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",   # 开启 VAD
        "vad_kwargs": {"max_single_segment_time": 30000}, # 强制每段最长 30s
        "device": DEVICE,
        "ncpu": THREADS,
        "punc_model": "ct-punc"  # 显式指定标点符号模型
    }
    
    model = AutoModel(**model_kwargs)
    
    print(f"✅ 模型加载完成，耗时: {time.time() - start_time:.2f}s")
    return model

def process_audio(model, audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 文件 {audio_file} 不存在")
        return None

    print(f"🎙️ 正在处理音频: {os.path.basename(audio_file)}...")
    print(f"📁 文件路径: {audio_file}")
    
    # 获取文件大小
    file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
    print(f"📊 文件大小: {file_size:.2f} MB")
    
    start_time = time.time()

    # 执行推理 Pipeline
    print("\n🔄 开始识别...")
    res = model.generate(
        input=audio_file,
        cache={},
        language="auto",  # 自动检测语言 (zh, en, yue, ja, ko)
        use_itn=True,     # 开启逆文本标准化 (例如: "一百" -> "100")
        batch_size_s=60,  # 动态批处理：每批处理 60秒 的音频数据
        merge_vad=True,   # 将切碎的 VAD 片段合并成整句
    )
    
    inference_time = time.time() - start_time
    
    print(f"\n✅ 识别完成，耗时: {inference_time:.2f}秒")
    
    # 结果解析
    if res:
        # res 可能是列表或字典，需要分别处理
        if isinstance(res, list):
            if len(res) > 0:
                result_item = res[0]
                if isinstance(result_item, dict):
                    raw_text = result_item.get("text", "")
                    print(f"\n📝 原始输出: {repr(raw_text[:200])}...")
                    
                    # 后处理
                    text = rich_transcription_postprocess(raw_text)
                    print(f"📝 后处理输出: {repr(text[:200])}...")
                    
                    # 移除标签
                    text = remove_sensevoice_tags(text)
                    print(f"📝 移除标签后: {repr(text[:200])}...")
                    
                    # 移除 emoji
                    text = remove_emoji(text)
                    print(f"📝 最终输出: {repr(text[:200])}...")
                    
                    return text
                else:
                    text = str(result_item) if result_item else ""
                    text = remove_sensevoice_tags(text)
                    text = remove_emoji(text)
                    return text
        elif isinstance(res, dict):
            raw_text = res.get("text", "")
            print(f"\n📝 原始输出: {repr(raw_text[:200])}...")
            
            text = rich_transcription_postprocess(raw_text)
            text = remove_sensevoice_tags(text)
            text = remove_emoji(text)
            return text
        else:
            text = str(res)
            text = remove_sensevoice_tags(text)
            text = remove_emoji(text)
            return text
    else:
        return "未检测到有效语音"

if __name__ == "__main__":
    # 处理指定的音频文件
    audio_file = "/Users/zhengyidi/AutoVoice/recordings/20251217_151202.webm"
    
    print("="*60)
    print("🚀 单独使用 SenseVoice 处理音频文件")
    print("="*60)
    
    # 1. 加载模型
    model = setup_model()
    
    # 2. 处理音频
    print("\n" + "="*60)
    result = process_audio(model, audio_file)
    
    if result:
        print("\n" + "="*60)
        print("📄 最终识别结果:")
        print("="*60)
        print(result)
        print("="*60)
        
        # 保存结果到文件
        output_file = os.path.join(
            os.path.dirname(audio_file),
            f"{os.path.splitext(os.path.basename(audio_file))[0]}_sensevoice_only.txt"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"音频文件: {os.path.basename(audio_file)}\n")
            f.write(f"处理方式: 单独使用 SenseVoice\n")
            f.write(f"\n识别结果:\n{result}\n")
        print(f"\n💾 结果已保存到: {output_file}")
    else:
        print("\n❌ 未识别到有效内容")

