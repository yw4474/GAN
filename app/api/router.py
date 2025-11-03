# app/api/router.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import zipfile, time

from app.services.gan_service import GANSampler

router = APIRouter()

gan = GANSampler(weights_path="weights/gan_mnist_gen.pt", device="cpu")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/gan/generate")
def gan_generate(n: int = 4):
    try:
        n = max(1, min(int(n), 16))  # 安全上限
        images = gan.sample_base64_pngs(n)
        return {"count": len(images), "images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gan/generate-files")
def gan_generate_files(
    n: int = Query(4, ge=1, le=64),
    out_dir: str = Query("weights/gan_samples")
):
    try:
        paths = gan.sample_png_files(n=n, out_dir=out_dir, prefix="gan")
        return JSONResponse({"count": len(paths), "dir": str(Path(out_dir).resolve()), "files": paths})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gan/generate-zip")
def gan_generate_zip(
    n: int = Query(8, ge=1, le=64),
    out_dir: str = Query("weights/gan_samples"),
    zip_dir: str = Query("weights/gan_exports")
):
    try:
        paths = gan.sample_png_files(n=n, out_dir=out_dir, prefix="gan")
        Path(zip_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_path = Path(zip_dir) / f"gan_samples_{ts}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=Path(p).name)

        return FileResponse(
            path=str(zip_path.resolve()),
            media_type="application/zip",
            filename=zip_path.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
