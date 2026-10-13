# LipCN 2.3.0

- 方案一在识别完成后先显示，立即可编辑、复制和输入；方案二、三随后补齐，保留用户修改。
- 明确选择“语音优先”或“纯口型”；纯口型和英文不请求或使用麦克风。
- 只加载当前方式需要的识别器，其他模型按需加载；目标文件缺失不释放原模型，加载失败回滚语言并尝试恢复原模型，可重试。
- `lipcn` 统一启动当前三语言桌面；新增只读 `doctor`、固定版本/哈希校验/可恢复的 `setup`。
- 可迁移 macOS Apple Silicon 应用构建脚本，随包携带 Python 和依赖，模型单独准备。未进行 Apple 公证。
- `lipcn collect` 本地无声样本采集、数据隔离审计和原始错误报告；本版未更换默认权重，未宣称无声准确率提高。

The first result is usable before wording finishes, all available candidates are editable, input mode is explicit, loading is on demand, and language-switch failures restore the previous selection. Unified launch/setup/diagnostics and a relocatable macOS build simplify installation. New local silent-data tools support future independent accuracy evaluation; this release makes no silent-accuracy claim.

## Verification scope

- Repository: **1,310 passed, 24 skipped**. Skips cover platform-specific or opt-in model/hardware checks.
- Real local wording model: [four synthetic text measurements](validation/2.3.0-wording.json). Option 1: 0.02–15.14 ms; all options: 0.57–9.40 s. Includes one cold wording-model load; excludes recognition and capture.
- Installed desktop: native window/three editors rendered, speech and visual loading completed, Chinese script and English selection exercised; camera and microphone remained closed.
- Startup was measured during concurrent packaging and low disk space; no generalized startup-speed claim is made. Optional background model work no longer queues ahead of the first sentence.
- Package checks: relocation, offline diagnostics, native imports and signature verification are distinct from real model/device validation. See [build instructions](../scripts/README.md).
- No new human recording, real cross-app insertion, independent silent test set or retraining was performed. [Silent protocol and historical evidence](../research/SILENT_EVALUATION.md).
