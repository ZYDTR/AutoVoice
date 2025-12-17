import os
import time
import torch
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ================= 配置区域 =================
# 模型 ID，会自动从 ModelScope 下载
MODEL_ID = "iic/SenseVoiceSmall"

# 设备选择策略
DEVICE = "cpu" 
THREADS = 4 

# 说话人识别配置
ENABLE_SPEAKER_DIARIZATION = True  # 设置为 True 启用说话人区分
SPK_MODEL = "cam++"  # 说话人识别模型

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

def setup_model():
    print(f"🔄 正在初始化模型 (Device: {DEVICE})...")
    if ENABLE_SPEAKER_DIARIZATION:
        print(f"📢 已启用说话人区分功能 (模型: {SPK_MODEL})")
    
    start_time = time.time()
    
    # AutoModel 初始化
    model_kwargs = {
        "model": MODEL_ID,
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",   # 开启 VAD
        "vad_kwargs": {"max_single_segment_time": 30000}, # 强制每段最长 30s
        "device": DEVICE,
        "ncpu": THREADS,
        "punc_model": "ct-punc"  # 显式指定标点符号模型，避免 punc_res 错误
    }
    
    # 如果启用说话人识别，添加 spk_model
    if ENABLE_SPEAKER_DIARIZATION:
        model_kwargs["spk_model"] = SPK_MODEL
        # spk_kwargs 可以用于配置说话人识别参数
        # 例如：spk_kwargs={"threshold": 0.5}  # 说话人相似度阈值
        print(f"   └─ 已自动加载标点符号模型（说话人识别需要）")
    
    model = AutoModel(**model_kwargs)
    
    print(f"✅ 模型加载完成，耗时: {time.time() - start_time:.2f}s")
    return model

def process_audio(model, audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 文件 {audio_file} 不存在")
        return None

    print(f"🎙️ 正在处理音频: {os.path.basename(audio_file)}...")
    start_time = time.time()

    # 执行推理 Pipeline
    res = model.generate(
        input=audio_file,
        cache={},
        language="auto",  # 自动检测语言 (zh, en, yue, ja, ko)
        use_itn=True,     # 开启逆文本标准化 (例如: "一百" -> "100")
        batch_size_s=60,  # 动态批处理：每批处理 60秒 的音频数据
        merge_vad=True,   # 将切碎的 VAD 片段合并成整句
    )
    
    inference_time = time.time() - start_time
    
    # 结果解析
    if res:
        # res 可能是列表或字典，需要分别处理
        if isinstance(res, list):
            if len(res) > 0:
                result_item = res[0]
                if isinstance(result_item, dict):
                    text = rich_transcription_postprocess(result_item.get("text", ""))
                    # 移除 emoji
                    text = remove_emoji(text)
                    # 检查是否包含说话人信息
                    speaker_info = result_item.get("spk", None)
                    if speaker_info:
                        print(f"📢 检测到说话人信息: {speaker_info}")
                    return {"text": text, "speaker": speaker_info, "raw": result_item}
                else:
                    text = rich_transcription_postprocess(result_item if result_item else "")
                    # 移除 emoji
                    text = remove_emoji(text)
                    return {"text": text, "speaker": None, "raw": result_item}
            else:
                return None
        elif isinstance(res, dict):
            text = rich_transcription_postprocess(res.get("text", ""))
            # 移除 emoji
            text = remove_emoji(text)
            speaker_info = res.get("spk", None)
            if speaker_info:
                print(f"📢 检测到说话人信息: {speaker_info}")
            return {"text": text, "speaker": speaker_info, "raw": res}
        else:
            text = str(res)
            # 移除 emoji
            text = remove_emoji(text)
            return {"text": text, "speaker": None, "raw": res}
    else:
        return None

def format_result_with_speaker(result, audio_file):
    """格式化带说话人信息的结果"""
    if not result:
        return "未检测到有效语音"
    
    text = result.get("text", "")
    speaker_info = result.get("speaker", None)
    raw_data = result.get("raw", {})
    
    output_lines = []
    output_lines.append(f"音频文件: {os.path.basename(audio_file)}\n")
    output_lines.append("="*60 + "\n")
    
    # 如果有说话人信息，格式化输出
    if speaker_info:
        output_lines.append("📢 说话人区分结果:\n")
        output_lines.append("-"*60 + "\n")
        
        # speaker_info 可能是列表或字典，需要根据实际格式处理
        if isinstance(speaker_info, list):
            for idx, spk in enumerate(speaker_info):
                if isinstance(spk, dict):
                    spk_id = spk.get("spk_id", f"Speaker_{idx}")
                    timestamp = spk.get("timestamp", "")
                    output_lines.append(f"说话人 {spk_id}: {timestamp}\n")
                else:
                    output_lines.append(f"说话人 {idx}: {spk}\n")
        elif isinstance(speaker_info, dict):
            for spk_id, info in speaker_info.items():
                output_lines.append(f"说话人 {spk_id}: {info}\n")
        else:
            output_lines.append(f"说话人信息: {speaker_info}\n")
        
        output_lines.append("\n")
    
    # 转录文本
    output_lines.append("识别结果:\n")
    output_lines.append("-"*60 + "\n")
    output_lines.append(text + "\n")
    
    # 如果有原始数据中的时间戳信息，也输出
    if isinstance(raw_data, dict):
        timestamp = raw_data.get("timestamp", None)
        if timestamp:
            output_lines.append("\n时间戳信息:\n")
            output_lines.append(f"{timestamp}\n")
    
    return "".join(output_lines)

if __name__ == "__main__":
    # 1. 加载模型
    model = setup_model()
    
    # 2. 处理 recordings 目录下的所有音频文件
    recordings_dir = "/Users/zhengyidi/AutoVoice/recordings"
    
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
                
                result = process_audio(model, audio_path)
                
                if result:
                    print("\n" + "="*20 + " 识别结果 " + "="*20)
                    formatted_result = format_result_with_speaker(result, audio_path)
                    print(formatted_result)
                    print("="*50)
                    
                    # 保存结果到文件
                    output_file = os.path.join(recordings_dir, f"{os.path.splitext(audio_file)[0]}_transcription.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_result)
                    print(f"💾 结果已保存到: {output_file}\n")
            
            print(f"\n✅ 所有文件处理完成！")
        
        # 说明：
        # 1. 说话人区分功能需要启用 spk_model="cam++"
        # 2. 输出结果中会包含说话人ID和时间戳信息
        # 3. 如果音频中只有一个说话人，可能不会显示说话人区分信息
        # 4. 多说话人场景下，每个说话人的语音会被标记并区分

