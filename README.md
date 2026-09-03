# LegacyAgent

A domain-adapted legal speech intelligence system I'm building on U.S. 
Supreme Court oral argument audio: fine-tuned ASR (automatic speech 
recognition) first, then verbatim transcription, then structured 
hearing minutes.

## V1 — ASR Fine-Tuning (Complete)

For V1, I fine-tuned Qwen3-ASR-1.7B on Supreme Court oral argument 
audio from the Qualified Immunity doctrine cluster, using QLoRA — a 
4-bit NF4 quantized base with LoRA adapters (r=32) via Hugging Face's 
PEFT library, applied only to the attention projections.

### Results (5 held-out cases, 1163 segments, identical WER scoring across all rows)

| Model | Overall WER | Entity-span WER |
|---|---|---|
| Whisper-small (reference) | 9.85% | — |
| Qwen3-ASR base | 7.71% | 30.67% |
| Qwen3-ASR + contextual biasing (free) | 7.57% | 27.27% |
| Qwen3-ASR + QLoRA fine-tune | 7.36% | 22.99% |
| Qwen3-ASR + QLoRA + int8 quantization | 7.22% | 21.39% |

Fine-tuning cut entity-span WER by 25.0% relative to base, and overall 
WER by 4.5% relative. Post-training int8 quantization didn't hurt 
either metric — so the gains survive being shrunk down for cheaper 
deployment.

### Model weights

I've got the LoRA adapter hosted on Hugging Face: 
https://huggingface.co/rir-i/legacyagent-qwen3-asr-lora

### Repo structure

- v1_asr_finetune/ — notebooks 01-08, the full pipeline
- eval/ — frozen entity vocabulary and evaluation protocol
- manifests/ — frozen train/val/test splits
- results/ — WER results per stage (baseline, biasing, fine-tune, quantization)

### Reproducing this

I ran everything on Google Colab (you'll need a GPU for the 4-bit 
QLoRA fine-tune). Each notebook is Python and pins its own 
dependencies in the first few cells — PyTorch, `transformers==5.13.0`, 
`peft`, `bitsandbytes`, `librosa`, `jiwer`. There's no repo-wide 
requirements.txt yet.

Audio isn't included here (see `.gitignore`) — the manifests in 
`manifests/` expect 16kHz mono WAV segments on Google Drive at 
`/content/drive/MyDrive/legacyagent/raw_audio/<case_id>.wav`, sourced 
from the `walkerdb/supreme_court_transcripts` Oyez mirror. Mount your 
own Drive at that path, then run notebooks 01-08 in order.

The dataset (Oyez transcripts + audio) is CC-BY-NC, non-commercial use 
only. The code here is MIT-licensed — see LICENSE.
