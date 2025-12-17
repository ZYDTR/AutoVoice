from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="录音上传服务")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建 recordings 目录
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "录音上传服务运行中"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    接收音频文件并保存到 recordings 目录
    """
    try:
        # 记录请求时间
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{request_time}] 收到上传请求 - 文件名: {file.filename}, Content-Type: {file.content_type}")
        
        # 读取文件内容
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"文件大小: {file_size} bytes ({file_size / 1024:.2f} KB)")
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1] or ".webm"
        filename = f"{timestamp}{file_extension}"
        
        # 保存文件的完整路径
        file_path = os.path.join(RECORDINGS_DIR, filename)
        absolute_path = os.path.abspath(file_path)
        
        # 写入文件
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"文件保存成功 - 本地绝对路径: {absolute_path}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "文件上传成功",
                "filename": filename,
                "file_path": file_path,
                "absolute_path": absolute_path,
                "file_size": file_size,
                "uploaded_at": request_time
            }
        )
        
    except Exception as e:
        error_msg = f"上传失败: {str(e)}"
        logger.error(f"错误: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


