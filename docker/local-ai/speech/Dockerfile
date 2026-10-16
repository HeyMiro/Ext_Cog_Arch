# Hey MiRo local speech service: speech-to-text (faster-whisper), text-to-
# speech (Kokoro) and end-of-turn detection (smart-turn), behind OpenAI-
# compatible HTTP endpoints. Models are NOT in the image: they are read from
# the heymiro-models volume (docker/local-ai/download_models.bash).
#
#   docker compose -f docker/local-ai/compose.yaml build speech

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	HF_HUB_OFFLINE=1 \
	HF_HUB_DISABLE_TELEMETRY=1 \
	MODELS_DIR=/models

# espeak-ng: Kokoro's fallback grapheme-to-phoneme for unknown words
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/speech/requirements.txt
RUN pip install --root-user-action=ignore -r /opt/speech/requirements.txt && \
	pip install --root-user-action=ignore \
		https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# ctranslate2 finds cuBLAS/cuDNN from torch's nvidia wheels
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY server.py /opt/speech/server.py
WORKDIR /opt/speech
EXPOSE 8090
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
	CMD curl -fsS http://localhost:8090/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
