# LipCN · 中文桌面运行说明 / Mandarin desktop setup

本目录是当前桌面入口。它复用仓库内的 `lipflow/` 和 `research/` 组件，提供中文语音优先、英文口型识别、简繁输出、三个表达方案、右 Command 切换和分阶段启动。内部 Python 包名保留兼容性；项目与窗口名称为 **LipCN**。

This is the current desktop entry. It uses components from `lipflow/` and `research/`, providing Mandarin speech-first recognition, English lip reading, simplified/traditional output, three wording options, a Right Command toggle and staged startup. Internal Python package names remain compatible; the project and window are named **LipCN**.

## 1. 环境 / Environment

需要 Apple Silicon Mac、macOS 14+、Python 3.11 或 3.12，以及相应的摄像头/麦克风权限。当前为源码发布；不把开发者的 Python 环境或模型权重放进 Git。另有可迁移运行时的打包脚本，模型仍单独准备。

Requires an Apple Silicon Mac, macOS 14+, Python 3.11 or 3.12, and the relevant camera/microphone permissions. Source and reproducible packaging scripts are provided; Python environments and model weights are not committed to Git. Models are prepared separately.

在仓库根目录运行 / From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[desktop]'
```

基础依赖来自根目录 `pyproject.toml`；本目录 requirements 增加 faster-whisper、CTranslate2、ONNX Runtime、OpenCC 和 SciPy。这里只安装一个 OpenCV 提供包 `opencv-contrib-python`，避免多个包同时覆盖 `cv2`。

Base dependencies come from the root `pyproject.toml`; the local requirements add faster-whisper, CTranslate2, ONNX Runtime, OpenCC and SciPy. Only one OpenCV provider, `opencv-contrib-python`, is selected to avoid multiple packages overwriting `cv2`.

## 2. 准备本地模型 / Prepare local models

启动不会自动下载模型。请先阅读各来源许可，再把模型放到下面的位置，或通过环境变量指向已有文件。项目代码的 MIT 许可不替代模型与数据的许可，详见 [NOTICE](../../NOTICE)。

Startup does not download models. Read each source’s terms, then put the assets in the locations below or point environment variables at existing files. The MIT code license does not replace model and dataset terms; see [NOTICE](../../NOTICE).

| 组件 / Component | 默认位置 / Default location | 来源与要求 / Source and requirements |
| --- | --- | --- |
| 中文语音 / Mandarin ASR | `~/.cache/lipcn/models/mandarin/faster-whisper-large-v3-turbo/` | [faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo), revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; CTranslate2 格式，至少包含 `model.bin`, `config.json`, `tokenizer.json` / CTranslate2 format, including these files |
| 本地表达整理 / Local wording | `~/.cache/lipcn/models/mandarin/Qwen3-1.7B-4bit/` | [Qwen3-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit), revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`; 完整 MLX 模型目录 / Complete MLX model directory |
| 口型基座 / Visual base model | `~/.cache/lipcn/models/mandarin/model_avg_cncvs_2_3_cnvsrc.pth` | [CNVSRC2025Baseline](https://huggingface.co/ReflectionL/CNVSRC2025Baseline), revision `b16f238d0df860da7e3b9834f959780b1d388f44`; checkpoint SHA-256 `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c` |
| 口型配置与词表 / Visual config and vocabulary | `~/.cache/lipcn/CNVSRC2025/` | [CNVSRC2025](https://github.com/liu12366262626/CNVSRC2025), revision `e5c4454016ba4eef9e586e77dd58e8981bb5c3e1`; 保留 `VSR/conf` 与 `VSR/datamodule` 布局 / Preserve the source layout |
| 本地口型适配参数 / Local visual adapter | `~/.cache/lipcn/models/mandarin/round8-encoder-adapter.pth` | 使用与上述基座兼容的本地 adapter；本仓库不分发开发者的 Round8 参数 / Supply a compatible local adapter; the developer’s Round8 adapter is not redistributed |
| 脸部跟踪 / Face tracking | `~/.cache/lipcn/models/face_landmarker.task`（CLI 默认 / CLI default） | [MediaPipe Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| 英文口型 / English visual model | `~/.cache/lipcn/models/english/vsr/` | [LRS3_V_WER19.1](https://huggingface.co/Amanvir/LRS3_V_WER19.1), revision `c0b413f56b164565865a63f00774ee55a9738cc5`; `model.json` 与 / and `model.pth` |
| 英文解码语言模型 / English decoding language model | `~/.cache/lipcn/models/english/lm/` | [lm_en_subword](https://huggingface.co/Amanvir/lm_en_subword), revision `f4e39b6d0904b2a900c53a2fff0377796c797bdc`; `model.json` 与 / and `model.pth` |

只准备语音模型也可以使用语音输入；缺少口型基座、词表或兼容适配参数时，口型回退不可用。默认快速文字整理不使用表达模型，三个方案可能重复；原有 MLX 润色可通过 `LIPCN_WORDING_MODE=model` 开启。`setup` 仍会准备含可选表达模型的完整资源组。要复现完整口型配置，另见[研究工具与模型说明](../../research/README.md)。

Speech input can work with just the ASR model. Lip fallback is unavailable without the visual checkpoint, vocabulary and compatible adapter. Fast formatting is the default, does not load the wording model, and may repeat options. Enable the previous MLX cleanup with `LIPCN_WORDING_MODE=model`. Setup still prepares the full resource group including the optional wording model. See the [research tools and model notes](../../research/README.md) for the visual setup.

推荐统一检查和准备模型： / Use the unified checks and model setup:

```bash
.venv/bin/lipcn doctor --mode speech
.venv/bin/lipcn setup --mode speech --download
.venv/bin/lipcn setup --mode english --download --accept-research-license
.venv/bin/lipcn doctor --mode all --json
```

`doctor` 不打开设备、不加载神经模型、不联网。下载按固定版本与 SHA-256 检查，临时文件验证通过后才替换目标；已验证文件可复用。中英文视觉研究模型的下载另需阅读来源许可并明确选择 `--accept-research-license`；本地 adapter 仍需自行提供。

`doctor` opens no devices, loads no neural models and uses no network. Setup verifies pinned versions and SHA-256 before atomically replacing a target file; verified files are reused. Mandarin and English research-model downloads require reviewing their source licenses and explicitly selecting `--accept-research-license`; the local adapter must still be supplied separately.

英文模型约 1.2 GB；需同时准备 VSR 和 LM 的四个文件。语言切换不会下载文件，也不会同时保留中英文视觉模型。简繁转换使用离线 OpenCC，无需额外神经网络模型。

The English assets total about 1.2 GB and require all four VSR/LM files. Switching languages does not download files or keep both visual models resident. Offline OpenCC handles Chinese script conversion without an additional neural model.

## 3. 启动与路径配置 / Launch and path configuration

```bash
lipcn
# 源码目录也可使用 / Or, from the checkout:
./desktop/mandarin/run.sh
```

`lipcn` 现在直接打开本页的三语言桌面窗口，`lipflow` 保留原有兼容入口。脚本默认使用仓库 `.venv/bin/python`；识别过程保持离线。

`lipcn` now opens the current three-language desktop; `lipflow` retains the legacy entry. The script uses the checkout’s `.venv/bin/python`, and recognition stays offline.

可以先设置以下变量，避免复制已有模型：

Set these variables before launch to reuse existing model files:

| 变量 / Variable | 作用 / Purpose |
| --- | --- |
| `LIPCN_FACE_MODEL` | 脸部跟踪文件 / Face-tracker file |
| `LIPCN_CAMERA` | 摄像头名称/标识，默认自动选择本机相机 / Camera name or ID; defaults to automatic local-camera selection |
| `LIPCN_PYTHON` | Python 可执行文件 / Python executable |
| `LIPCN_CACHE_DIR` | 缓存根目录，默认 `~/.cache/lipcn` / Cache root |
| `LIPCN_MANDARIN_MODELS` | 模型根目录，默认 `<cache>/models/mandarin` / Model root |
| `LIPCN_ENGLISH_MODELS` | 英文模型根目录，含 `vsr/` 与 `lm/`，默认 `<cache>/models/english` / English model root containing `vsr/` and `lm/` |
| `LIPCN_ASR_MODEL` | 语音模型目录 / ASR model directory |
| `LIPCN_FORMATTER_MODEL` | MLX 表达模型目录 / Wording model directory |
| `LIPCN_WORDING_MODE` | 默认 `fast` 使用保守规则；`model` 启用较慢的本地 MLX 润色 / `fast` uses conservative rules (default); `model` opts into slower local MLX cleanup |
| `LIPCN_CNVSRC_CHECKPOINT` | 口型基座文件 / Visual checkpoint file |
| `LIPCN_CNVSRC_SOURCE` | CNVSRC2025 配置/词表来源目录 / Config and vocabulary source |
| `LIPCN_LIP_ADAPTER` | 兼容的本地口型 adapter / Compatible local visual adapter |
| `LIPCN_LIVE_STATE` | 仅数字/布尔状态文件目录，默认 `<cache>/runtime/mandarin` / Diagnostic status directory |
| `LIPCN_PROFILE_DIR` | 独立配置目录，默认 `~/Library/Application Support/LipCN` / Separate profile directory |

```bash
export LIPCN_ASR_MODEL="/path/to/faster-whisper-large-v3-turbo"
export LIPCN_FORMATTER_MODEL="/path/to/Qwen3-1.7B-4bit"
./desktop/mandarin/run.sh
```

不要把模型、录音、视频、运行状态、个人配置或 `.env` 提交到 Git。本仓库已经忽略常见模型扩展名、模型目录与本地运行目录。

Keep models, recordings, videos, runtime state, personal profiles and `.env` files out of Git. Common model extensions, model directories and runtime directories are ignored.

## 4. 使用与验证 / Use and verification

- 右上角语言选择提供“简体中文”“繁體中文”“English”，每次启动默认简体中文。简繁共用中文识别，只转换最终结果字形；复制和输入保持所选字形。 / The top-right language selector offers Simplified Chinese, Traditional Chinese and English, defaulting to Simplified Chinese on each launch. Chinese modes share one recognizer; final output, copy and insertion use the selected writing system.
- 输入方式提供“语音优先”和“纯口型”，记住上次选择；纯口型不请求麦克风权限，也不调用语音识别。 / Input mode offers Speech first and Lip only, remembers the choice, and never requests microphone permission or runs ASR in Lip only.
- 方案一先显示，其余方案随后补齐；可在文本框直接修改，复制和输入使用修改后的内容，后续生成不覆盖已编辑文本。 / Option 1 appears first. Edit available text directly; copy and insertion use your edits, which later generation preserves.
- English 只使用摄像头和英文口型模型，不启动麦克风、不使用中文语音识别，也不翻译。 / English uses only the camera and English visual model, without microphone capture, Mandarin ASR or translation.
- 简繁切换无需重载模型。中英文切换先检查目标文件，加载失败时恢复原选择并尝试恢复原模型；可点击“重试加载”。加载、采集与识别时不可切换。 / Chinese script switches reuse one model. Visual language switches check target files first; failures restore the prior selection and attempt to restore its model. Retry is available; selection is disabled during loading, capture or inference.
- 点击开始，或允许输入监控权限后按右 Command；再次按下结束，Esc 取消。 / Click Start, or use Right Command after allowing Input Monitoring; press again to finish, Esc to cancel.
- 摄像头和麦克风仅在开始后采集；复制/输入必须由用户选择。 / Capture starts only on request; copying and insertion require an explicit choice.
- 结束本句后停止录音和识别帧收集；摄像头仅作实时预览，直到结果完成。取消、隐藏、最小化或关闭会请求停止并清空画面。预览期间降低脸部跟踪频率，录制本句仍逐帧跟踪。 / Finish ends utterance audio and recognition-frame collection; the camera continues live preview until results complete. Cancel, hide, minimize or close requests shutdown and clears the images. Preview-only tracking is throttled; recording still tracks every frame.
- 从终端启动时，macOS 权限可能显示为终端或 Python；按系统设置中的实际名称授权。 / When launched from a terminal, macOS may assign permissions to that terminal or Python; use the identity shown in System Settings.
- 启动只准备当前方式需要的识别器，备用口型与表达模型按需加载，避免在第一句前排队。 / Startup prepares the selected recognizer; fallback visual and wording models load only on demand, avoiding optional initialization ahead of the first sentence.
- 本版本不自动读取历史句子、不自动训练，也不自动把文字提交到其他应用。 / This entry does not automatically read sentence history, train on the user, or submit text in other apps.

无硬件桌面回归测试 / Headless desktop regression tests:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q desktop/mandarin/tests
```

完整仓库测试可能跳过缺少模型、其他平台或需要主动授权的真实粘贴测试；测试结果不等于真实无声识别准确率。

Full repository tests may skip missing models, other platforms or opt-in real-paste tests. Passing tests does not establish accuracy on real silent speech.
