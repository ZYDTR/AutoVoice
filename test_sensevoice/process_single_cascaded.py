#!/usr/bin/env python3
"""
使用级联系统处理单个音频文件
"""
import os
import sys
from run_cascaded_system import (
    setup_cascaded_models,
    process_audio_cascaded,
    format_cascaded_result
)

if __name__ == "__main__":
    # 处理指定的音频文件
    audio_file = "/Users/zhengyidi/AutoVoice/recordings/20251205 234222-BF444D4E_part_004.m4a"
    
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 文件 {audio_file} 不存在")
        sys.exit(1)
    
    print("="*60)
    print("🚀 使用级联系统处理音频文件")
    print("="*60)
    print(f"📁 文件: {os.path.basename(audio_file)}")
    
    # 获取文件大小
    file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
    print(f"📊 文件大小: {file_size:.2f} MB")
    print("="*60)
    
    # 1. 加载模型
    print("\n🔄 正在加载模型...")
    paraformer_model, sensevoice_model = setup_cascaded_models()
    
    # 2. 处理音频
    print("\n" + "="*60)
    try:
        final_results = process_audio_cascaded(
            audio_file, 
            paraformer_model, 
            sensevoice_model
        )
        
        # 格式化输出
        formatted_result = format_cascaded_result(final_results, os.path.basename(audio_file))
        
        print("\n" + "="*60)
        print("📄 识别结果:")
        print("="*60)
        print(formatted_result)
        print("="*60)
        
        # 保存结果到文件
        output_file = os.path.join(
            os.path.dirname(audio_file),
            f"{os.path.splitext(os.path.basename(audio_file))[0]}_cascaded_transcription.txt"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_result)
        print(f"\n💾 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

