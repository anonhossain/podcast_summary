# import subprocess
# import sys
# import shutil
# from pathlib import Path

# # ---------------- CONFIG ----------------
# OLLAMA_MODEL = "gpt-oss:20b"
# HF_MODEL_ID = "openai/whisper-large-v3-turbo"
# HF_TARGET_DIR = Path("models/whisper-large-v3-turbo")
# # ----------------------------------------


# def command_exists(cmd):
#     return shutil.which(cmd) is not None


# def run(cmd):
#     subprocess.run(cmd, check=True)


# # ---------- OLLAMA SECTION ----------
# def check_ollama_installed():
#     if not command_exists("ollama"):
#         print("❌ Ollama is not installed.")
#         print("👉 Install from: https://ollama.com")
#         sys.exit(1)


# def ollama_model_exists(model):
#     result = subprocess.run(
#         ["ollama", "list"],
#         capture_output=True,
#         text=True
#     )
#     return model in result.stdout


# def ensure_ollama_model(model):
#     print(f"🔍 Checking Ollama model: {model}")
#     if ollama_model_exists(model):
#         print(f"✅ Ollama model '{model}' already installed.")
#     else:
#         print(f"⬇️ Pulling Ollama model '{model}'...")
#         run(["ollama", "pull", model])
#         print(f"✅ Ollama model '{model}' installed.")


# # ---------- HUGGING FACE SECTION ----------
# def ensure_hf_dependencies():
#     try:
#         import huggingface_hub
#     except ImportError:
#         print("⬇️ Installing Hugging Face Hub...")
#         run([sys.executable, "-m", "pip", "install", "huggingface_hub"])


# def ensure_whisper_model(model_id, target_dir):
#     if target_dir.exists() and any(target_dir.iterdir()):
#         print(f"✅ Whisper model already exists at {target_dir}")
#         return

#     print(f"⬇️ Downloading Whisper model: {model_id}")
#     from huggingface_hub import snapshot_download

#     snapshot_download(
#         repo_id=model_id,
#         local_dir=target_dir,
#         local_dir_use_symlinks=False
#     )

#     print("✅ Whisper model downloaded successfully.")


# # ---------- MAIN ----------
# def main():
#     print("\n=== SETUP STARTED ===\n")

#     check_ollama_installed()
#     ensure_ollama_model(OLLAMA_MODEL)

#     ensure_hf_dependencies()
#     ensure_whisper_model(HF_MODEL_ID, HF_TARGET_DIR)

#     print("\n🎉 SETUP COMPLETED SUCCESSFULLY 🎉\n")


# if __name__ == "__main__":
#     main()


# import subprocess
# import shutil
# import sys
# from pathlib import Path
# from huggingface_hub import snapshot_download
# import asyncio

# # ---------------- CONFIG ----------------
# OLLAMA_MODEL = "gpt-oss:20b"
# HF_MODEL_ID = "openai/whisper-large-v3-turbo"
# HF_DIR = Path("models/whisper-large-v3-turbo")
# # ---------------------------------------


# # ---------- UTIL ----------
# def command_exists(cmd: str) -> bool:
#     return shutil.which(cmd) is not None


# # ---------- OLLAMA ----------
# def ollama_installed() -> bool:
#     return command_exists("ollama")


# def ollama_model_exists(model: str) -> bool:
#     result = subprocess.run(
#         ["ollama", "list"],
#         capture_output=True,
#         text=True
#     )
#     return model in result.stdout


# def ensure_ollama_model():
#     if not ollama_installed():
#         raise RuntimeError("Ollama is not installed")

#     if ollama_model_exists(OLLAMA_MODEL):
#         return "Ollama model already exists"

#     subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
#     return "Ollama model downloaded"


# # ---------- WHISPER ----------
# def ensure_whisper_model():
#     if HF_DIR.exists() and any(HF_DIR.iterdir()):
#         return "Whisper model already exists"

#     snapshot_download(
#         repo_id=HF_MODEL_ID,
#         local_dir=HF_DIR,
#         local_dir_use_symlinks=False
#     )
#     return "Whisper model downloaded"


# # ---------- PARALLEL SETUP ----------
# async def setup_models_parallel():
#     loop = asyncio.get_running_loop()

#     ollama_task = loop.run_in_executor(None, ensure_ollama_model)
#     whisper_task = loop.run_in_executor(None, ensure_whisper_model)

#     results = await asyncio.gather(
#         ollama_task,
#         whisper_task,
#         return_exceptions=True
#     )

#     return {
#         "ollama": str(results[0]),
#         "whisper": str(results[1]),
#     }



import subprocess
import shutil
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIG =================
OLLAMA_MODEL = "gpt-oss:20b"
HF_MODEL_ID = "openai/whisper-large-v3-turbo"
HF_DIR = Path("models/whisper-large-v3-turbo")
# =========================================


# =============== UTIL =====================
def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# =============== OLLAMA ===================
def ollama_installed() -> bool:
    return command_exists("ollama")


def ollama_model_exists(model: str) -> bool:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    return model in result.stdout


def ensure_ollama_model():
    print(f"🦙 Ollama started at {time.strftime('%X')}")

    if not ollama_installed():
        raise RuntimeError("Ollama is not installed")

    if ollama_model_exists(OLLAMA_MODEL):
        print("🦙 Ollama model already exists")
        return "Ollama model already exists"

    subprocess.run(
        ["ollama", "pull", OLLAMA_MODEL],
        check=True
    )

    print(f"🦙 Ollama finished at {time.strftime('%X')}")
    return "Ollama model downloaded"


# =============== WHISPER ==================
def ensure_whisper_model():
    print(f"🎙️ Whisper started at {time.strftime('%X')}")

    if HF_DIR.exists() and any(HF_DIR.iterdir()):
        print("🎙️ Whisper model already exists")
        return "Whisper model already exists"

    HF_DIR.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=HF_DIR,
        local_dir_use_symlinks=False
    )

    print(f"🎙️ Whisper finished at {time.strftime('%X')}")
    return "Whisper model downloaded"


# ========== PARALLEL SETUP =================
async def setup_models_parallel():
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=2) as executor:
        ollama_future = loop.run_in_executor(
            executor, ensure_ollama_model
        )
        whisper_future = loop.run_in_executor(
            executor, ensure_whisper_model
        )

        results = await asyncio.gather(
            ollama_future,
            whisper_future,
            return_exceptions=True
        )

    return {
        "ollama": str(results[0]),
        "whisper": str(results[1]),
    }


# ========== CLI SUPPORT ====================
if __name__ == "__main__":
    print("\n🚀 Starting parallel model setup...\n")
    result = asyncio.run(setup_models_parallel())
    print("\n✅ Setup completed\n")
    print(result)