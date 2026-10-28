# vLLM 공식 이미지는 라이선스 문제로 음성 의존성(librosa, soundfile)을 넣지 않는다.
# Whisper 전사 서버용으로 같은 버전의 vllm[audio] extra만 추가한다.
ARG N_NOTE_VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129
FROM ${N_NOTE_VLLM_IMAGE}
ARG N_NOTE_VLLM_VERSION=0.31.0
RUN python3 -m pip install --no-cache-dir "vllm[audio]==${N_NOTE_VLLM_VERSION}"
