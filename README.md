# Bert-VITS2

VITS2 Backbone with bert
## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应该参阅代码自己学习如何训练。
### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
---
## 环境搭建
### 创建 conda 环境
```
conda create -n Bert-VITS2 python=3.8
conda activate Bert-VITS2
```
### 安装依赖
```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```
### 构建 monotonic_align
```
cd monotonic_align && python setup.py build_ext --inplace && cd ..
```
### 下载 bert 模型
*[下载 bert 模型](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)，并放置到 bert/chinese-roberta-wwm-ext-large 目录中*

## 使用
- 首先将训练用的 wav 格式音频文件放置到 workspace/raw_audio 目录下。
- 文件命名规则为 <说话人>_<任意数字>.wav，如 paimeng_1.wav、paimeng_2.wav。
- 可以同时放置多个说话人的多个音频文件。
### 提取人声
```
python scripts/denoise_audio.py
```
### whisper 识别
```
python scripts/transcribe_audio.py
```
- 默认使用 whisper 的 large 模型，12G 以下显存可以使用 `--whisper_size medium` 加载 medium 模型。
- 首次运行会从 Hugging Face 拉取模型权重文件。
```
python scripts/transcribe_audio.py --whisper_size medium
```
### 重采样
```
python resample.py
```
### 标注并划分训练集和验证集
```
python preprocess_text.py
```
### 生成 .bert.pt
```
python bert_gen.py
```
### 训练
- 运行如下命令开始训练，使用 `-m` 参数指定训练名称，会在 ./logs/ 目录下创建同名目录存放权重文件。
```
python train_ms.py -m test
```
- 如果要继续之前中断的训练，可以使用 `--resume` 参数。
```
python train_ms.py -m test --resume
```
### 推理 WebUI
- 运行如下命令会自动在浏览器中打开 `http://127.0.0.1:7860`，使用 `-m` 参数指定模型文件
```
python webui.py -m ./logs/test/G_2000.pth
```
- Google Colab 等场合可以使用 `--share` 参数，生成可内网穿透的链接。
```
python webui.py -c ./logs/test/config.json -m ./logs/test/G_2000.pth --share
```
### 推理 API
- 运行如下命令后会启动 API 并监听 `http://0.0.0.0:8000`。
- 可访问 `http://127.0.0.1:8000/docs` 查看 API 手册。
```
python api.py -c ./logs/test/config.json -m ./logs/test/G_2000.pth
```
