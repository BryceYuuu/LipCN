# 中文识别与候选确认（实验性）

这次扩展提供两种不同的中文输入路径。**当前不应把中文纯唇读用于无人确认的日常输入。**

| 模式 | 输入 | 模型 | 已知边界 |
| --- | --- | --- | --- |
| 中文无声 | 摄像头，无需声音 | 作者发布的 CMLR 中文视觉模型及中文字符语言模型 | 研究用途；3360 余个汉字词表；没有英文混合识别能力；摄像头和跨数据集效果未达到可用标准 |
| 中文低声 | 摄像头质量检查 + 麦克风音频识别 | faster-whisper / Whisper large-v3-turbo，CPU int8 | 这是音频 ASR 加视觉质量门控，不是训练过的中文 AVSR 融合；必须有可听见的低声或耳语；结果需确认 |

## 安装与使用

先按主 README 完成基础安装。Mac 用 `./setup.sh`，Windows 用 `setup.ps1`。基础安装提供摄像头人脸检测模型。

中文纯唇读模型来自 [原作者模型库](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#model-zoo)。下载前阅读[源项目许可](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/LICENSE)及 [CMLR 数据集用途说明](https://www.vipazoo.cn/CMLR.html)。此适配不授予商业使用权。权重不随本仓库、安装包或 PR 分发。

```bash
uv sync --extra chinese
uv run --extra chinese lipflow install-chinese --accept-research-license
uv run lipflow run --language zh --cleanup basic --confidence-policy review
```

模型下载约 400 MB，安装器验证已核实档案的 SHA-256，只写入配置和权重两个指定成员。源档案如有变化会报错，不能关闭校验绕过。

中文低声输入（不需要安装 CMLR 权重）：

```bash
uv sync --extra chinese-whisper
uv run --extra chinese-whisper lipflow run --language zh --input-mode whisper --cleanup basic
```

第一次启动低声模式会下载约 1.6 GB 的 Whisper 模型。此后识别在本地运行。Mac 使用 CPU int8；不是 MLX/Apple GPU 音频识别。麦克风只在按住输入键及收尾期间打开。完全不出声时不要使用此模式。中文音频结果统一转换为简体；此转换不是跨语言翻译。

菜单提供识别语言、忠实/润色模式、候选策略和输入模式；菜单中的这些设置需要重启生效。CLI 参数优先于保存的设置。Mac 现有 Whisper 开关继续控制英语 AVSR；中文低声模式使用上述独立输入模式。

## 候选确认

默认 `--confidence-policy review`。识别后浮窗显示最多三个不同结果，包含纠错建议、原始识别和其他候选；按 1、2、3 选择，Esc 取消重说。数字快捷键仅在浮窗有焦点时使用，不安装全局数字键拦截。选定后恢复原输入应用并检查目标；检查失败时仅复制，提示手动粘贴。

Mac 会检查应用 PID 和原输入控件。Windows 当前只检查窗口 HWND，尚未实现控件/光标级 UI Automation 检查。用户在等待期间移动同一控件内的光标仍无法完全检测。

质量检查使用人脸可见比例、原图嘴宽像素及嘴部画面的亮度/对比度。失败时提示靠近摄像头、改善光线或正对摄像头。无脸、无唇动、未知字符和明显重复输出也会提示重说。所有阈值都是保守启发式，不能诊断所有识别错误。

可选自动策略：

```bash
uv run lipflow run --language zh --cleanup basic --confidence-policy auto --min-margin 0.5
```

只有至少两个带有限分数的候选、足够的长度归一化分数差、CTC 与首选候选一致、画面质量通过且纠错没有风险提示时才自动输入。**分数差不是正确概率，尚未用真实摄像头用户数据校准。** 单候选的中文低声 ASR 始终需要确认。

本地历史中保存原候选、总分、token 数、路由理由、纠错建议及风险提示，便于后续独立校准。不会把未确认文本加入输入历史/上下文。

## 统一纠错边界

本地、Claude、Ollama 均经过相同的输出检查。数字、日期、时间、金额、配置的姓名/术语、否定表达发生变化时，必须确认；忠实模式还检查新增候选外词语和大幅修改。润色改变措辞时需要确认。标点和可确定等价的英文数字格式允许正常处理。

检查是规则约束，不是语义等价的证明。未配置姓名、同数量否定词移动和复杂金额表述仍可能漏检。小模型也可能拒绝建议而退回基础格式化。原始结果保留在浮窗、历史和 “Copy raw recognition” 菜单中；该菜单复制原文，不会盲目修改用户已编辑的文档。

`--cleanup basic` 完全避免调用文本 LLM。原有 `auto` 后端仍可能在环境有 Anthropic 凭证时选择云端 Claude。

## 中文训练与文本处理

中文练习使用 60 条本项目编写的日常句子，每轮抽取 24 条。中文练习片段位于 `clips/onboarding/zh/`，中文脸部参数位于 `models/zh/vsr_face.pth`，与已有英语参数分开。中文训练按模型字符词表生成目标，拒绝不支持的英文和阿拉伯数字，而不是静默丢弃中文。已有练习训练流程继续进行留出集检查。

中文模式不加载英语个人语言模型；英语 `train-lm` 仍是英语工具。中文历史可用于字符级短语检索和重排。中文上下文仅从明确的收件人/私聊标题标记提取姓名，不从普通正文猜测姓名。自定义中文词汇可在句子内匹配，不会使用英语唇形规则强行替换中文姓名。中文自动纠错样本采用保守字符对齐，Windows 仍没有自动纠错采集。

重新训练需要足够的真实用户片段。此 PR 没有证明 24 句即可改善个人中文准确率，也没有证明跨光线/日期泛化。

## 视频与复现

完整人脸视频：

```bash
uv run lipflow file face-video.mp4 --language zh --cleanup basic
```

已对齐的 96×96 嘴部片段必须指定 `--mouth-roi`，否则人脸检测会误处理这些片段：

```bash
uv run lipflow file mouth-video.mp4 --language zh --mouth-roi --cleanup basic
```

可使用 `scripts/evaluate_chinese.py` 输出字符错误率（CER）、延迟和候选分数，详见 [验证记录](VALIDATION.md)。本版本没有附带他人的录制片段或未经授权的数据集。
