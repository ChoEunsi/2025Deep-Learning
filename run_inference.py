import torch
import torchaudio
import os
#from seperator.seperator.seperator import seperator  # 경로는 실제 환경에 맞게 조정 필요
sources = ["drums", "saxophone", "vocals", "other"] #source 지정하는 코드가 seperator안에 있으면 중복 조심
#import sys
#sys.path.append("C:/Users/.../Desktop/.../seperator")


MODEL_PATH = "best.th"
INPUT_AUDIO = "input.wav"
OUTPUT_DIR = "separated_stems"
SAMPLE_RATE = 44100  # 일반적으로 44.1kHz 오디오 사용

print("모델 초기화 중...")
model = seperator(sources=sources)
state = torch.load(MODEL_PATH, map_location="cpu")

# 일부 체크포인트는 'model.state_dict()' 없이 바로 state가 dict인 경우도 있음
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state)
model.eval()
print("모델 로드 완료")

# input 오디오 처리
print(f"오디오 불러오는 중: {INPUT_AUDIO}")
waveform, sr = torchaudio.load(INPUT_AUDIO)

if sr != SAMPLE_RATE:
    print(f"샘플레이트 {sr} → {SAMPLE_RATE}로 변환 중...")
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    waveform = resampler(waveform)

# batch 차원 추가
waveform = waveform.unsqueeze(0)

# task
print("오디오 분리 중...")
with torch.no_grad():
    output = model(waveform)  # output shape: (1, stems, channels, samples)

# save
os.makedirs(OUTPUT_DIR, exist_ok=True)
stem_names = model.sources if hasattr(model, "sources") else [f"stem_{i}" for i in range(output.shape[1])]

print(f"분리된 stem 저장 중... → {OUTPUT_DIR}")
for i, name in enumerate(stem_names):
    stem = output[0, i]  # shape: (channels, samples)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    torchaudio.save(out_path, stem.cpu(), SAMPLE_RATE)
    print(f"저장 완료: {out_path}")

print("분리 완료")
