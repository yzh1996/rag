"""
文件上传API
"""
import shutil
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.params import Depends
from fastapi.responses import JSONResponse
import logging
from config.settings import settings
from api.dependencies import get_vector_store_dep
from core.vector_store import VectorStoreManager
from core.document_loader import DocumentLoader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["文件上传"])


def process_file_background(file_path: str, metadata: dict, vector_store: VectorStoreManager):
    """后台处理文件并添加到向量库"""
    try:
        loader = DocumentLoader()
        docs = loader.load_file(file_path, metadata)
        if docs:
            vector_store.add_documents(docs)
            logger.info(f"后台处理完成: {file_path}，添加 {len(docs)} 个片段")
        else:
            logger.warning(f"文件无有效内容: {file_path}")
    except Exception as e:
        logger.error(f"后台处理文件失败: {file_path}, 错误: {e}")


@router.post("/file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    上传单个文件
    """
    # 验证文件类型
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"不支持的文件格式: {file_ext}")

    # 生成唯一文件名
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    save_path = settings.UPLOAD_DIR / safe_filename

    # 保存文件
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    # 元数据
    metadata = {
        "category": category,
        "original_filename": file.filename,
        "upload_time": str(Path(save_path).stat().st_ctime)
    }

    # 后台处理添加到向量库
    background_tasks.add_task(
        process_file_background,
        str(save_path),
        metadata,
        vector_store
    )

    return JSONResponse({
        "status": "success",
        "message": f"文件 {file.filename} 上传成功，正在后台处理",
        "file_id": unique_id,
        "filename": safe_filename
    })


@router.post("/batch")
async def upload_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    category: str = Form("general"),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    批量上传文件
    """
    results = []
    for file in files:
        try:
            file_ext = Path(file.filename).suffix.lower()
            allowed = {'.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls'}
            if file_ext not in allowed:
                results.append({"filename": file.filename, "status": "skipped", "reason": "不支持的文件格式"})
                continue

            unique_id = uuid.uuid4().hex[:8]
            safe_filename = f"{unique_id}_{file.filename}"
            save_path = settings.UPLOAD_DIR / safe_filename

            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            metadata = {
                "category": category,
                "original_filename": file.filename,
            }
            background_tasks.add_task(process_file_background, str(save_path), metadata, vector_store)

            results.append({"filename": file.filename, "status": "success", "file_id": unique_id})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "reason": str(e)})

    return JSONResponse({"results": results})