# LipCN · 中文桌面运行说明 / Mandarin desktop setup

本目录是当前中文桌面入口。它复用仓库内的 `lipflow/` 和 `research/` 组件，新增语音优先、三个表达方案、右 Command 切换和分阶段启动。内部 Python 包名保留兼容性；项目与窗口名称为 **LipCN**。

This is the current Mandarin desktop entry. It uses components from `lipflow/` and `research/`, adding speech-first recognition, three wording options, a Right Command toggle and staged startup. Internal Python package names remain compatible; the project and window are named **LipCN**.

## 1. 环境 / Environment

需要 Apple Silicon Mac、macOS 14+、Python 3.11 或 3.12，以及相应的摄像头/麦克风权限。当前为源码发布；没有把开发者的 Python 环境、模型权重或 `.app` 安装包放进 Git。

Requires an Apple Silicon Mac, macOS 14+, Python 3.11 or 3.12, and the relevant camera/microphone permissions. This is a source release; the developer’s Python environment, model weights and `.app` bundle are not included in Git.

在仓库根目录运行 / From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . -r desktop/mandarin/requirements.txt
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
| 脸部跟踪 / Face tracking | 仓库内 / In checkout: `models/face_landmarker.task` | [MediaPipe Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |

只准备语音模型也可以使用语音输入；缺少口型基座、词表或兼容适配参数时，口型回退不可用。缺少表达模型时会使用保守的基本整理，三个方案可能重复。要复现完整口型配置，另见[研究工具与模型说明](../../research/README.md)。

Speech input can work with just the ASR model. Lip fallback is unavailable without the visual checkpoint, vocabulary and compatible adapter. Without the wording model, conservative basic formatting remains available and options may repeat. See the [research tools and model notes](../../research/README.md) for the visual setup.

脸部跟踪文件单独下载，不必运行安装旧版应用的 `setup.sh`：

Download the face tracker separately; running the legacy application’s `setup.sh` is not required:

```bash
mkdir -p models
curl --fail --location \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  --output models/face_landmarker.task
```

如使用 Hugging Face CLI，语音和表达模型可以按固定版本下载：

With the Hugging Face CLI, download the speech and wording models at the pinned revisions:

```bash
.venv/bin/hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo \
  --revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf \
  --local-dir "$HOME/.cache/lipcn/models/mandarin/faster-whisper-large-v3-turbo"
.venv/bin/hf download mlx-community/Qwen3-1.7B-4bit \
  --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
  --local-dir "$HOME/.cache/lipcn/models/mandarin/Qwen3-1.7B-4bit"
```

## 3. 启动与路径配置 / Launch and path configuration

```bash
./desktop/mandarin/run.sh
```

脚本默认使用仓库 `.venv/bin/python`，并让模型加载保持离线。全局 `lipcn` 与兼容的 `lipflow` CLI 运行的是保留的旧核心入口；要打开本页的新中文窗口，请使用上面的脚本。

The script uses the checkout’s `.venv/bin/python` and keeps model loading offline. The global `lipcn` and compatibility `lipflow` commands run the retained legacy core; use the script above for the current Mandarin window.

可以先设置以下变量，避免复制已有模型：

Set these variables before launch to reuse existing model files:

| 变量 / Variable | 作用 / Purpose |
| --- | --- |
| `LIPCN_PYTHON` | Python 可执行文件 / Python executable |
| `LIPCN_CACHE_DIR` | 缓存根目录，默认 `~/.cache/lipcn` / Cache root |
| `LIPCN_MANDARIN_MODELS` | 模型根目录，默认 `<cache>/models/mandarin` / Model root |
| `LIPCN_ASR_MODEL` | 语音模型目录 / ASR model directory |
| `LIPCN_FORMATTER_MODEL` | MLX 表达模型目录 / Wording model directory |
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

- 点击开始，或允许输入监控权限后按右 Command；再次按下结束，Esc 取消。 / Click Start, or use Right Command after allowing Input Monitoring; press again to finish, Esc to cancel.
- 摄像头和麦克风仅在开始后采集；复制/输入必须由用户选择。 / Capture starts only on request; copying and insertion require an explicit choice.
- 从终端启动时，macOS 权限可能显示为终端或 Python；按系统设置中的实际名称授权。 / When launched from a terminal, macOS may assign permissions to that terminal or Python; use the identity shown in System Settings.
- 启动显示阶段与耗时；语音就绪后可先采集，剩余模型继续加载。 / Startup displays stages and time; recording can begin once speech is ready while other models continue loading.
- 本版本不自动读取历史句子、不自动训练，也不自动把文字提交到其他应用。 / This entry does not automatically read sentence history, train on the user, or submit text in other apps.

无硬件桌面回归测试 / Headless desktop regression tests:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q desktop/mandarin/tests
```

完整仓库测试可能跳过缺少模型、其他平台或需要主动授权的真实粘贴测试；测试结果不等于真实无声识别准确率。

Full repository tests may skip missing models, other platforms or opt-in real-paste tests. Passing tests does not establish accuracy on real silent speech.
