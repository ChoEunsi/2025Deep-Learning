#!/usr/bin/env python3
"""
integrate_sax.py

musdb18-hq (train+test 150곡)에 saxophone.wav(44.1 kHz) 추가하고 5-stem 합친 mixture.wav를 재생성
- Track 폴더 이름 그대로 복사
- sax 파일이 150 이상이면 앞 150개만 사용, 부족하면 남은 트랙은 무음 패딩 : 무음 넣을 수는 없으니 이거 나오면 sax track 개수 늘려서 다시 실행
- saxophone 연주되는 구간이 더 긴 경우, 무음 구간은 대체로 앞부분에 있으므로 앞부분에서 잘라내고 mixture, saxophone이 더 짧으면 뒷부분에 padding
"""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import librosa
import shutil

# ──── 경로 설정 ──────────────
MUSDB_ROOT = Path("/home/kangin/demucs_dataset/musdb18_hq")
SAX_DIR    = Path("/home/kangin/demucs_dataset/sax_wav_44k")
OUT_ROOT   = Path("/home/kangin/demucs_dataset/musdb18_hq_with_sax")
# ────────────────────────────

def load_wav(p: Path, sr: int) -> np.ndarray:
    audio, orig_sr = sf.read(p)
    if orig_sr != sr:
        audio = librosa.resample(audio.T, orig_sr=orig_sr, target_sr=sr).T
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    return audio

def fit_to_len(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) < length:
        pad = np.zeros((length - len(audio), 2), dtype=audio.dtype)
        return np.vstack([pad, audio])
    else:
        return audio[-length:]

def integrate(track_src: Path, track_dst: Path, sax_path: Optional[Path]):
    track_dst.mkdir(parents=True, exist_ok=True)

    # 1) mixture 참조
    mix_src = track_src / "mixture.wav"
    mix_audio, sr = sf.read(mix_src)
    length = len(mix_audio)

    # 2) bass/drums/vocals/other 복사
    for stem in ("bass", "drums", "vocals", "other"):
        src = track_src / f"{stem}.wav"
        if src.exists():
            shutil.copy(src, track_dst / f"{stem}.wav")

    # 3) saxophone 처리 (없으면 무음)
    if sax_path:
        sax = load_wav(sax_path, sr)
        sax = fit_to_len(sax, length)
    else:
        sax = np.zeros((length, 2), dtype=np.float32)
        print(f"{track_src.name}에 대응하는 sax 파일 없음, 무음 사용")

    sf.write(track_dst / "saxophone.wav", sax, sr, subtype="PCM_16")

    # 4) mixture 생성
    mix = sax.copy()
    for stem in ("bass", "drums", "vocals", "other"):
        st = track_dst / f"{stem}.wav"
        if st.exists():
            a = load_wav(st, sr)
            mix += a[:length]

    peak = np.max(np.abs(mix))
    if peak > 0.999:
        mix /= peak

    sf.write(track_dst / "mixture.wav", mix, sr, subtype="PCM_16")
    print(f"{track_src.name} 통합 완료")

def main():
    # 0) 출력 폴더 준비
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) sax 파일 리스트
    sax_all = sorted(SAX_DIR.glob("*.wav"))
    print(f"🔍 sax 파일 총 {len(sax_all)}개 발견")

    # 2) 트랙 폴더 리스트 (train + test)
    tracks = []
    for split in ("train","test"):
        d = MUSDB_ROOT/split
        tracks += sorted([p for p in d.iterdir() if p.is_dir()])
    print(f"musdb 트랙 총 {len(tracks)}개 처리 예정")

    # 3) 1대1 매핑, 통합
    for idx, track in enumerate(tracks):
        sax = sax_all[idx] if idx < len(sax_all) else None
        dst = OUT_ROOT / track.relative_to(MUSDB_ROOT)
        print(f"[{idx+1}/{len(tracks)}] {track.name} ← {sax.name if sax else '무음'}")
        integrate(track, dst, sax)

if __name__ == "__main__":
    main()
