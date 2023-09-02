import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from fastapi import FastAPI, Body
from fastapi.responses import Response
from webui import infer
import uvicorn
from scipy.io import wavfile
from io import BytesIO
import ffmpeg

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps,)
    with torch.no_grad():
        x_tst=phones.to(device).unsqueeze(0)
        tones=tones.to(device).unsqueeze(0)
        lang_ids=lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids,bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        return audio

app = FastAPI()

@app.post("/v1/tts")
def tts(speaker: str = Body(..., title="说话人", embed=True),
        text: str = Body(..., title="文本", embed=True),
        sdp_ratio: float = Body(0.2, embed=True),
        noise: float = Body(0.5, embed=True),
        noisew: float = Body(0.6, embed=True),
        length: float = Body(1.0, embed=True),
        format: str = Body('wav', embed=True)):
    with torch.no_grad():
        audio = infer(text, sdp_ratio, noise, noisew, length, speaker)
    torch.cuda.empty_cache()
    wav = BytesIO()
    wavfile.write(wav, hps.data.sampling_rate, audio)
    wav.seek(0)
    if format == "mp3":
        process = (
            ffmpeg
                .input("pipe:", format='wav', channel_layout="mono")
                .output("pipe:", format='mp3', audio_bitrate="320k")
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        out, _ = process.communicate(input=wav.read())
        return Response(out, headers={"Content-Disposition": 'attachment; filename="audio.mp3"'}, media_type="audio/mpeg")
    else:
        return Response(wav.read(), headers={"Content-Disposition": 'attachment; filename="audio.wav"'}, media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="./logs/test/G_2000.pth", help="path of your model")
    parser.add_argument("-c", "--config", default="./workspace/config.json", help="path of your config file")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None,skip_optimizer=True)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
