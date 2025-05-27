import os
import soundfile as sf
import numpy as np

def mix_stems_to_mixture(track_folder, stems=None, output_filename="mixture.wav"):
    if stems is None:
        stems = ['vocals', 'drums', 'bass', 'guitar', 'saxophone', 'other']

    stem_paths = [os.path.join(track_folder, f"{stem}.wav") for stem in stems]
    audio_list = []
    sr = None

    for path in stem_paths:
        if os.path.exists(path):
            audio, sr = sf.read(path)
            if audio.ndim == 1:
                # mono → stereo 변환
                audio = np.stack([audio, audio], axis=1)
            audio_list.append(audio)
        else:
            print(f"[경고] {path} 없음. 무시하고 진행.")

    if not audio_list:
        raise RuntimeError("혼합할 오디오가 없습니다.")

    min_len = min([a.shape[0] for a in audio_list])
    audio_list = [a[:min_len] for a in audio_list]

    mixture = np.sum(audio_list, axis=0)

    max_val = np.max(np.abs(mixture))
    if max_val > 1.0:
        mixture /= max_val

    output_path = os.path.join(track_folder, output_filename)
    sf.write(output_path, mixture, samplerate=sr)
    print(f"[완료] mixture.wav 저장됨: {output_path}")


if __name__ == "__main__":
    mix_stems_to_mixture(r"C:\Users\조은시\Desktop\GIST\3학년 1학기\딥러닝 - 이규빈\최종프로젝트\sample_dataset\mega_augmented_ds\mega_augmented_ds\mydataset\t1",
        stems=['bass', 'drums', 'guitar', 'piano'] 
    )
