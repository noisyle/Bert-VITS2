# Bert-VITS2

VITS2 Backbone with bert
## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应该参阅代码自己学习如何训练。
### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 一个小提示，这个模型用FP16跑会炸炸炸炸炸，从0开始训练也有概率炸炸炸炸炸，因此最好是load一个VITS原版的底模,并丢弃emb_g

## 安装依赖
```
conda create -n Bert-VITS2 python=3.8
conda activate Bert-VITS2
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

conda install -c conda-forge ffmpeg

cd monotonic_align && python setup.py build_ext --inplace && cd ..
```

## 降噪
```
python scripts/denoise_audio.py
```

## whisper识别
```
python scripts/long_audio_transcribe.py
```

## 重采样
```
python resample.py
```

## 标注并划分训练集和验证集
```
python preprocess_text.py
```

## 生成.bert.pt
```
python spec_gen.py
```

## 训练
```
python train_ms.py -c ./configs/config.json -m as
```

## 推理
```
python webui.py -c ./configs/config.json -m ./logs/as/G_8000.pth
```
