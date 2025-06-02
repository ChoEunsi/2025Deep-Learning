"""
download_tracks.py --n_tracks 600 : 맨 뒤에 숫자는 추출하기 위해 스캔할 트랙 개수. 모든 트랙에 색소폰이 있는게 아니므로 원하는 색소폰 트랙 개수보다 크게 설정해야 함.
Slakh2100 축소판에서 mix.flac + all_src.mid 를 N개 트랙만큼 내려받아 디렉터리에 저장. (wav 파일이 아니라 어떤 파일을 받아올지 리스트를 정하는 것과 같음)
hf_slakh_subset/TrackXXXX/ 디렉터리에 저장
"""
import argparse, re
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

REPO  = "DreamyWanderer/Slakh2100-FLAC-Redux-Reduced"
RTYPE = "dataset"
OUT   = Path("hf_slakh_subset")

def main(n_tracks: int):
    OUT.mkdir(exist_ok=True)

    # 1) slakh2100 축소판에 있는 모든 all_src.mid 경로
    midi_paths = [
        p for p in list_repo_files(REPO, repo_type=RTYPE)
        if p.endswith("all_src.mid")
    ]

    # 2) (split, TrackID) 튜플 추출
    tracks = []
    for p in midi_paths:
        m = re.search(r"(train|val|test)/(Track\d+)", p)
        if m:
            split, track = m.groups()
            tracks.append((split, track))

    # 3) 아직 안 받은 트랙만 N개 선택
    picked, seen = [], set()
    for split, track in sorted(tracks, key=lambda x: x[1]):   # Track00001 … 순
        if len(picked) >= n_tracks:
            break
        if (OUT / track).exists(): # 이미 있는 건 pass
            continue
        if track not in seen:
            picked.append((split, track))
            seen.add(track)

    print(f"Downloading {len(picked)} tracks (mix+MIDI each)")
    for split, track in tqdm(picked):
        for fname in ("mix.flac", "all_src.mid"):
            hf_hub_download(
                repo_id   = REPO,
                repo_type = RTYPE,
                filename  = f"data/{split}/{track}/{fname}",
                local_dir = OUT / track,
                resume_download = True
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tracks", type=int, default=800,
                    help="받을 트랙 수 (디폴트 800)")
    args = ap.parse_args()
    main(args.n_tracks)
