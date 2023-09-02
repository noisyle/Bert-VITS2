import os
import json
import torchaudio

config_file = "./configs/base.json"
raw_audio_dir = "./workspace/raw_audio/"
denoise_audio_dir = "./workspace/denoised_audio/"
filelist = list(os.walk(raw_audio_dir))[0][2]

# 2023/4/21: Get the target sampling rate
with open(config_file, 'r', encoding='utf-8') as f:
    hps = json.load(f)
target_sr = hps['data']['sampling_rate']

if not os.path.exists(denoise_audio_dir):
    os.mkdir(denoise_audio_dir)

for file in filelist:
    if file.lower().endswith(".wav"):
        os.system(f"demucs --two-stems=vocals {raw_audio_dir}{file} -o ./workspace")
        prefix = file.rsplit('.', 1)[0]
        wav, sr = torchaudio.load(f"./workspace/htdemucs/{prefix}/vocals.wav", frame_offset=0, num_frames=-1, normalize=True,
                                channels_first=True)
        # merge two channels into one
        wav = wav.mean(dim=0).unsqueeze(0)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        torchaudio.save(denoise_audio_dir + prefix + ".wav", wav, target_sr, channels_first=True)
