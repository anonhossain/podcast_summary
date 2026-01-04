from fastapi import FastAPI
from backend.views import api
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Audio Transcription and Summarization", 
    version="1.0",
    description="This is an audio transcription and summarization app powered by AI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to access the API, you can specify domains as well
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8080, reload=True) #localhost 10.10.13.7