import os
import time
import torch
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ================= 配置区域 =================
# 处理模式选择
PROCESSING_MODE = "direct"  # "direct" 或 "cascaded"
# - "direct": 直接使用单一模型（SenseVoice 或 Paraformer）
# - "cascaded": 级联模式（Paraformer 做 diarization + SenseVoice 识别文本）

# 模型选择：支持 SenseVoice 和 Paraformer（仅在 direct 模式下有效）
# SenseVoice: 不支持 speaker diarization
# Paraformer: 支持 speaker diarization
USE_MODEL = "sensevoice"  # "sensevoice" 或 "paraformer"

SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
PARAFORMER_MODEL = "paraformer-zh"

# 说话人识别配置（仅 Paraformer 支持，仅在 direct 模式下有效）
ENABLE_SPEAKER_DIARIZATION = False  # 设置为 True 启用说话人区分（需要 USE_MODEL="paraformer"）
SPK_MODEL = "cam++"  # 说话人识别模型

# 设备选择策略
# M1 Pro 建议：
# 对于 < 1分钟的短音频，或者追求稳定性，使用 "cpu" 是最佳选择，速度极快且无兼容性问题。
# 对于 > 10分钟的长音频批处理，可以尝试 "mps" (Metal Performance Shaders)。
# 如果遇到报错，请回退到 "cpu"。
DEVICE = "cpu" 
# DEVICE = "mps" # 取消注释以尝试 GPU 加速

# 线程数设置 (仅对 CPU 模式有效)
# M1 Pro 有 8 或 10 个核心，设置为 4-8 之间通常效率最高
THREADS = 4 

# ===========================================

def setup_model():
    # 根据配置选择模型
    if USE_MODEL == "paraformer":
        model_id = PARAFORMER_MODEL
        model_name = "Paraformer"
    else:
        model_id = SENSEVOICE_MODEL
        model_name = "SenseVoice"
    
    print(f"🔄 正在初始化模型: {model_name} (Device: {DEVICE})...")
    
    if ENABLE_SPEAKER_DIARIZATION:
        if USE_MODEL != "paraformer":
            print("⚠️ 警告: SenseVoice 模型不支持 speaker diarization，已自动禁用")
            enable_spk = False
        else:
            enable_spk = True
            print(f"📢 已启用说话人区分功能 (模型: {SPK_MODEL})")
    else:
        enable_spk = False
    
    start_time = time.time()
    
    # AutoModel 初始化
    model_kwargs = {
        "model": model_id,
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",   # 开启 VAD
        "vad_kwargs": {"max_single_segment_time": 30000}, # 强制每段最长 30s
        "device": DEVICE,
        "ncpu": THREADS,
        "punc_model": "ct-punc"  # 显式指定标点符号模型，避免 punc_res 错误
    }
    
    # 如果启用说话人识别，添加 spk_model（仅 Paraformer）
    if enable_spk:
        model_kwargs["spk_model"] = SPK_MODEL
        print(f"   └─ 已自动加载标点符号模型（说话人识别需要）")
        print(f"   ℹ️ 输出时将过滤掉 timestamp，只显示说话人 ID 和文本")
    
    model = AutoModel(**model_kwargs)
    
    print(f"✅ 模型加载完成，耗时: {time.time() - start_time:.2f}s")
    return model

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

def process_audio(model, audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 文件 {audio_file} 不存在")
        return

    print(f"🎙️ 正在处理音频: {os.path.basename(audio_file)}...")
    start_time = time.time()

    # 执行推理 Pipeline
    # generate() 内部流程:
    # 1. VAD 扫描音频，生成时间戳列表
    # 2. 根据时间戳切分音频 (Chunking)
    # 3. 对每个 Chunk 进行 SenseVoice 推理 (Inference)
    # 4. 合并结果 (Merging)
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
    # 输出可能包含说话人信息，需要分别处理
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
                        print(f"📢 检测到说话人信息")
                        # 格式化说话人信息（不显示 timestamp）
                        speaker_texts = []
                        if isinstance(speaker_info, list):
                            for spk in speaker_info:
                                if isinstance(spk, dict):
                                    spk_id = spk.get("spk_id", "Unknown")
                                    spk_text = spk.get("text", "") or spk.get("sentence", "")
                                    if spk_text:
                                        speaker_texts.append(f"说话人 {spk_id}: {spk_text}")
                        if speaker_texts:
                            print("\n".join(speaker_texts))
                    print(f"✅ 处理完成，耗时: {inference_time:.2f}s")
                    return text
                else:
                    text = rich_transcription_postprocess(result_item if result_item else "")
                    text = remove_emoji(text)
                    print(f"✅ 处理完成，耗时: {inference_time:.2f}s")
                    return text
        elif isinstance(res, dict):
            text = rich_transcription_postprocess(res.get("text", ""))
            # 移除 emoji
            text = remove_emoji(text)
            # 检查是否包含说话人信息
            speaker_info = res.get("spk", None)
            if speaker_info:
                print(f"📢 检测到说话人信息")
                # 格式化说话人信息（不显示 timestamp）
                speaker_texts = []
                if isinstance(speaker_info, list):
                    for spk in speaker_info:
                        if isinstance(spk, dict):
                            spk_id = spk.get("spk_id", "Unknown")
                            spk_text = spk.get("text", "") or spk.get("sentence", "")
                            if spk_text:
                                speaker_texts.append(f"说话人 {spk_id}: {spk_text}")
                if speaker_texts:
                    print("\n".join(speaker_texts))
            print(f"✅ 处理完成，耗时: {inference_time:.2f}s")
            return text
        else:
            text = str(res)
            text = remove_emoji(text)
            print(f"✅ 处理完成，耗时: {inference_time:.2f}s")
            return text
    else:
        return "未检测到有效语音"

if __name__ == "__main__":
    # 根据处理模式选择不同的处理方式
    if PROCESSING_MODE == "cascaded":
        # 级联模式：先 Paraformer 做 diarization，再用 SenseVoice 识别
        print("="*60)
        print("🚀 使用级联模式处理")
        print("="*60)
        
        # 导入级联系统模块
        try:
            from run_cascaded_system import (
                setup_cascaded_models,
                process_audio_cascaded,
                format_cascaded_result
            )
            
            # 加载模型
            paraformer_model, sensevoice_model = setup_cascaded_models()
            
            # 处理 recordings 目录下的所有音频文件
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
        
        except ImportError as e:
            print(f"❌ 导入级联系统模块失败: {str(e)}")
            print("请确保 run_cascaded_system.py 文件存在")
            import traceback
            traceback.print_exc()
    
    else:
        # 直接模式：使用单一模型
        print("="*60)
        print("🚀 使用直接模式处理")
        print("="*60)
        
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
                        print(result)
                        print("="*50)
                        
                        # 保存结果到文件
                        output_file = os.path.join(recordings_dir, f"{os.path.splitext(audio_file)[0]}_transcription.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"音频文件: {audio_file}\n")
                            f.write(f"识别结果:\n{result}\n")
                        print(f"💾 结果已保存到: {output_file}\n")
                
                print(f"\n✅ 所有文件处理完成！")
        
        # 结果解释：
        # 输出可能包含类似 <|zh|><|NEUTRAL|><|Speech|> 的标签
        # <|zh|>: 语言
        # <|NEUTRAL|>: 情感 (HAPPY, SAD, ANGRY, NEUTRAL)
        # <|Speech|>: 事件 (可能包含 BGM, Laughter 等)