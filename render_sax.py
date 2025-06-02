from pathlib import Path
import pretty_midi, subprocess, multiprocessing, tqdm, os, re, sys

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SRC = Path("hf_slakh_subset")
# DST = Path("sax_wav_16k")
# TARGET_SR = 16000
DST = Path("sax_wav_44k")
TARGET_SR = 44100

DST.mkdir(exist_ok=True)

def is_sax(instr):
    return 64 <= instr.program <= 67 or re.search(r"sax", instr.name, re.I)

def render(track_root):
    try:
        midi_path = next(track_root.rglob("all_src.mid"))
    except StopIteration:
        return

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    sax_instr = [ins for ins in pm.instruments if is_sax(ins)]
    if not sax_instr:
        return

    sax_pm = pretty_midi.PrettyMIDI()
    for ins in sax_instr:
        new_ins = pretty_midi.Instrument(program=ins.program)
        new_ins.notes = ins.notes
        sax_pm.instruments.append(new_ins)

    tmp_mid = track_root / "tmp_sax.mid"
    sax_pm.write(str(tmp_mid))

    out_wav = DST / f"{track_root.name}_sax.wav"
    
    if out_wav.exists():
        return
    
    # 리버브, 코러스 모두 있는 기본 버전
    # subprocess.run(
    #     ["fluidsynth", "-ni", SF2, str(tmp_mid),
    #      "-F", str(out_wav), "-r", str(TARGET_SR)],
    #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    # )
    
    # 리버브, 코러스 제거
    # subprocess.run(
    #     ["fluidsynth", "-ni", "-R", "0", "-C", "0", SF2, str(tmp_mid),
    #      "-F", str(out_wav), "-r", str(TARGET_SR)],
    #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    # )
    
    # 리버브 25%, 코러스 제거
    subprocess.run(
        ["fluidsynth", "-ni",
        "-o", "synth.reverb.active=1",
        "-o", "synth.reverb.level=0.25",   # 0.0~1.2 추천 범위
        "-o", "synth.chorus.active=0",
        SF2, str(tmp_mid),
        "-F", str(out_wav), "-r", str(TARGET_SR)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    tmp_mid.unlink()



if __name__ == "__main__":
    tracks = sorted(SRC.glob("Track*/"))

    n_workers = min(8, os.cpu_count())       # CPU 코어 수 : 최대 8
    with multiprocessing.Pool(n_workers) as p:
        list(tqdm.tqdm(
            p.imap_unordered(render, tracks),
            total=len(tracks),
            desc="Rendering sax"
        ))
    print("색소폰 WAV 생성 완료 →", DST)
