python3 ./whisper_llm_zh/decode.py \
  --max-duration 80 \
  --on-the-fly-feats True \
  --exp-dir ./whisper_llm_zh/exp_librispeech \
  --speech-encoder-path-or-name large-v2 \
  --llm-path-or-name Qwen/Qwen2-1.5B-Instruct \
  --epoch 10 --avg 1 \
  --manifest-dir data/fbank \
  --use-flash-attn True \
  --use-lora False --dataset librispeech
