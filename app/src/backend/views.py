from fastapi import APIRouter, HTTPException
from .setup import setup_models_parallel
import os
from fastapi import UploadFile, File
import shutil
from fastapi.responses import JSONResponse


api = APIRouter()


@api.post("/setup/models", tags=["Setup"])
async def setup_models():
    try:
        result = await setup_models_parallel()
        return {
            "status": "success",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os
import shutil
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.transcription import transcribe_audio_segmented

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FILES_DIR = os.path.join(BASE_DIR, "files")

@api.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload ANY audio/video file and receive timestamped transcription
    """

    input_path = os.path.join(FILES_DIR, file.filename)

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe
        segments = transcribe_audio_segmented(
            audio_path=input_path,
            segment_length=15
        )

        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "segments": segments
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()








#####################################################################





from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import os

from backend.orchestrator import PodcastOrchestrator

app = FastAPI(title="Podcast Shortener API")

orchestrator = PodcastOrchestrator()


@app.post("/process", response_class=StreamingResponse)
def process_media(file: UploadFile = File(...)):
    try:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_path = tmpdir / file.filename

            # Save upload INSIDE temp dir
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Run full pipeline
            output_path = orchestrator.process(
                input_file_path=str(input_path),
                work_dir=tmpdir,
            )

            # Stream output
            def file_iterator():
                with open(output_path, "rb") as f:
                    yield from f

            return StreamingResponse(
                file_iterator(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{output_path.name}"'
                },
            )

        # 🧹 TempDirectory auto-deletes EVERYTHING here

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
