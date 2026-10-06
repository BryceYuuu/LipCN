# Local maintainer-review fixes / 维护者反馈本地修复

Date / 日期：2026-10-03。Base / 基点：`be8faa5d8556c9a56bc80e78a64ea65502293cb1`。

## 中文

本次只修改本地文件并验证，没有创建提交、推送或更新合并申请。

- **英文默认行为**：`faithful` + 未选择候选策略时使用 `off`，直接输入纠错结果，不打开新增候选窗、不新增 CTC 解码或质量/分差/焦点拦截。明确选择 `review`/`auto` 才启用新增流程。CLI 优先于保存设置，中文保存的策略不会误开启英文候选确认。
- **英文纠错回归**：恢复 `bark → park`、`five → 5` 和自定义词/上下文姓名纠正，恢复原先英文测试预期。本地小模型原有候选词及改动数量限制仍保留。
- **中文和润色**：两者始终开启敏感改动保护及确认，不能被 `off`/`auto` 或误传的 `False` 绕过。本地、Claude、Ollama 检查一致，保留原文和待确认的建议。
- **上下文开关**：`use_context=False` 不调用完整 `capture()`，不读取标题、输入控件或内容。默认英文不需要查询目标；确认模式只保存应用 PID/窗口 HWND。
- **时间与重采样**：摄像头、麦克风、推理耗时和短句拼接间隔使用单调时钟。有效时间线继续选择最近帧；倒退、重复、缺失、非有限时间戳保持原采集顺序，不抛异常、不排序打乱图像/landmarks。每次搜索清除活动 decoder 的 `_mem_kv` 修复继续保留。
- **研究边界**：研究脚本、文档、汇总 JSON 和人工采集器集中在 `research/`，普通应用不再包含采集 CLI。历史数值 JSON 逐字节不变；移动代码后历史训练只能在原冻结源码中续训，避免混用实验身份。
- **工作流权限**：保护检查工作流明确限定为 `permissions: contents: read`。本次没有运行远端 CI。

## English

All changes and validation remain local. No commit, push, or PR update was made.

Faithful English defaults to direct cleaned input (`off`), with additional candidate routing and sensitive-edit checks enabled only by explicit opt-in. The original homophene, number-formatting and name corrections are restored, as are their original test expectations. The local model retains its existing candidate-word/edit limits. Mandarin and explicitly selected polish always require review and guard every cleanup backend. Disabling context prevents reading the focused field, title or text; reviewed input remembers only the target application/window identity.

Capture and microphone timestamps, inference elapsed time and phrase-join intervals use a monotonic clock. Valid timelines still select nearest frames; unusable timestamps preserve capture order without exceptions or frame/landmark reordering. Active decoder memory is cleared per search. Research tools and historical reports live under `research/`, separate from the default legacy application commands. The explicitly selected `desktop/mandarin/` LipCN research desktop entry reuses its model-loading and scoring helpers. Frozen numerical evidence is unchanged, and old training continuations require the original source checkout. The guard workflow explicitly grants only `contents: read`.

## Validation / 验证

- Final complete local suite: **368 passed, 13 skipped, 0 failed, 0 errors** (381 cases, 3.96 s).
- Tests exercise both frontends' real Python control flow, with fake camera, OS/UI, cleanup and paste boundaries. They verify default English cleanup-to-paste, mandatory Mandarin/polish review, language-specific preferences and CLI precedence, disabled context access, and quiet-mode initialization/fallback.
- Capture-loop and file-playback simulations include a one-hour backward wall-clock adjustment, microphone/video alignment, idle close and original ROI/landmark order. Reused encoder storage exercises English and AV decoder cache clearing.
- All **11** research CLI `--help` commands work from a temporary directory with `PYTHONPATH` removed; no model, data download or camera is started.
- `git diff --check`, Python compilation, workflow YAML permissions and local documentation links pass. Historical benchmark JSON is byte-identical (SHA-256 `5be0f9f2c7c07b19482ae127c52939a24c7dbfcdcfafd2b284a0cb72c75952a9`).
- Skips: **9 native Windows tests**, **2 optional Chinese model-weight tests**, **1 real English sample-video test** and **1 actual paste test**. New native Windows UI, real camera/microphone capture, real cross-app paste and real-model accuracy were not tested in this round. Windows control flow with simulated boundaries is not native Windows validation.

本地完整套件 **368 通过、13 跳过、0 失败、0 错误**。两端实际控制流通过模拟 OS/UI、模型及粘贴边界验证；不进行真实输入。11 个研究入口的帮助命令、编译、文档路径与 YAML 权限均检查通过。

跳过项明确保留：原生 Windows 9、可选中文权重 2、真实英文视频 1、实际粘贴 1。本轮没有重跑原生 Windows 界面、真实摄像头/麦克风、跨应用粘贴或模型准确率测试。此前公开数据研究数值属于历史结果，本轮代码回归测试不构成新的中文识别准确率证据。
