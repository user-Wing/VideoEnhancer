# VideoEnhancer 1.4.2 补帧模型集成设计

## 目标

正式启用 VideoEnhancer 1.4.2，将补帧模型统一迁移到 `models\Frame-Interpolation`，修复 RIFE 移动后无法扫描的问题，并贯通 RIFE、GMFSS Fortuna、GIMM-VFI 的 Python 后端、CLI 中转、插件界面和模型下载器。

## 范围与约束

- 只修改 `1.4.2` 版本源码和 `C:\PortableSoft\VideoEnhancer-CLI` 对应后端/部署内容。
- RIFE 位于 `models\Frame-Interpolation\RIFE`。
- GMFSS 位于 `models\Frame-Interpolation\GMFSS`。
- GIMM-VFI 位于 `models\Frame-Interpolation\GIMM-VFI`。
- GIMM-VFI 和 GMFSS 只正式支持 CUDA/PyTorch；不宣称支持 NCNN、ONNX 或原生 TensorRT。
- 转换上游发布的所有推理版本；不包含仅用于训练的预训练权重。
- 如果任一源文件或转换结果超过 1 GB，必须在下载或继续处理前停止并询问用户。
- 每个模型版本只发布一个组合权重文件；转换过程使用的源文件数量不设限制。
- 保持现有超分、BasicVSR++、FlashVSR、队列、暂停、停止、预览和处理顺序功能不变。
- 只做实现本需求所需的局部修改。

## 模型资产

### RIFE

保留现有 NCNN 子目录结构：

```text
Frame-Interpolation\RIFE\rife-v4.25\flownet.param
Frame-Interpolation\RIFE\rife-v4.25\flownet.bin
```

CUDA RIFE 权重允许作为 `.pth`、`.pt` 或 `.pkl` 文件放在 RIFE 目录及其子目录中。

### GIMM-VFI

转换 Hugging Face 上游发布的四个 VFI 推理 checkpoint：

```text
Frame-Interpolation\GIMM-VFI\GIMM-VFI-R.pth
Frame-Interpolation\GIMM-VFI\GIMM-VFI-R-LPIPS.pth
Frame-Interpolation\GIMM-VFI\GIMM-VFI-F.pth
Frame-Interpolation\GIMM-VFI\GIMM-VFI-F-LPIPS.pth
```

R 版本把相应 checkpoint 与 `raft-things.pth` 合并。F 版本把相应 checkpoint 与 `flowformer_sintel.pth` 合并。最终文件包含模型类型元数据和运行所需的全部 state dict，不依赖转换时的临时文件。

### GMFSS Fortuna

转换上游 Model Zoo 的三个推理包：

```text
Frame-Interpolation\GMFSS\GMFSS-Fortuna-Base.pth
Frame-Interpolation\GMFSS\GMFSS-Fortuna-Union.pth
Frame-Interpolation\GMFSS\GMFSS-Fortuna-Union-AnimeRun.pth
```

最终文件包含 `model_type` 元数据以及该版本实际使用的 `flownet`、`metricnet`、`feat_ext`、`fusionnet` 和可选 `rife` state dict。Base 不要求 `rife`，Union 系列必须包含 `rife`。

## 后端设计

`rve-backend.py` 继续使用统一的 `--interpolate_model` 参数，但在启动渲染前验证补帧模型路径、扩展名和后端组合。GIMM-VFI 或 GMFSS 与非 CUDA/PyTorch 后端组合时返回明确错误。

`InterpolateFactory` 根据组合权重的结构和元数据选择 RIFE、GIMM-R、GIMM-F 或 GMFSS。架构检测不得只依赖文件名。

GIMM-R 沿用现有适配器；GIMM-F 从已克隆的官方 GIMM-VFI 仓库引入其 FlowFormer 推理所需的最小代码，不导入训练、数据集或评估模块。两种 GIMM 共享现有帧缓存和任意插值时间点调用接口。

GMFSS 加载器先读取 `model_type`，再决定是否建立和加载 RIFE 子网络，修复 Base 权重被无条件索引 `rife` 的问题。缺少 CuPy 时允许使用现有 PyTorch soft-splat 回退并输出性能提示；不把缺少 CuPy 当成模型不可用。

## CLI 扫描与调用

`Program.cs` 定义唯一的补帧根目录 `models\Frame-Interpolation`：

- NCNN 只发现 `RIFE` 下同时存在 `.param` 和 `.bin` 的目录。
- CUDA 递归发现 RIFE、GMFSS 和 GIMM-VFI 中的 `.pth`、`.pt`、`.pkl` 权重。
- 放大模型扫描排除整个 `Frame-Interpolation`，避免补帧权重出现在超分列表中。
- 补帧显示名使用相对 `Frame-Interpolation` 的路径，保留架构前缀。
- 解析优先接受完整相对路径；仅当文件名唯一时才接受省略目录的模型名。
- `--list-interp-models --json` 是插件获取补帧模型的稳定接口。
- 版本常量、项目版本、构建脚本和用户代理统一更新为 `1.4.2`。

## 插件界面

沿用现有补帧模型下拉框，不增加新的架构选择控件。列表项显示 `RIFE/...`、`GMFSS/...`、`GIMM-VFI/...`：

- 选择 GMFSS 或 GIMM-VFI 时自动把补帧后端切换到 CUDA。
- 非 CUDA 环境下仍展示已安装模型，但禁用启动或给出明确状态提示，不静默选择其他模型。
- 选择 RIFE 时保留 NCNN、CUDA 和现有 TensorRT 行为；TensorRT 仅适用于后端真实支持的 RIFE 权重。
- 配置继续保存相对路径，旧的裸 RIFE 模型名在唯一匹配时仍可解析。
- 队列参数继续通过 `-interp-model`、`-interp-backend` 和 `-interp-factor` 传递。

## 下载器

远程模型清单允许 `Frame-Interpolation` 顶层分类。安装状态按远端相对路径映射到本地同一路径，归档解压根目录保持在 `models`，避免形成重复的 `Frame-Interpolation\Frame-Interpolation`。

下载页按 RIFE、GMFSS、GIMM-VFI 子分类显示；下载完成后刷新补帧模型列表。源文件归档和转换临时文件不作为最终可选模型显示。

远端 ModelScope 仓库若尚未包含转换后的权重，代码先完成路径与分类支持；是否上传模型资产属于外部写操作，需要单独得到用户授权。

## 错误处理

- 目录不存在时，列表返回空数组并给出新的标准路径提示。
- 模型路径越出 `models\Frame-Interpolation` 时拒绝相对路径解析；显式绝对路径仍按现有 CLI 规则处理。
- GMFSS/GIMM 权重缺少必要键时，在启动 FFmpeg 前失败并列出缺失键。
- 模型架构无法识别时返回可操作错误，不允许工厂返回 `None` 后产生二次异常。
- GIMM-F 依赖不完整时明确指出缺失模块或权重，不回退到 GIMM-R。
- 用户请求不受支持的后端组合时返回非零退出码。

## 测试与验证

按测试驱动方式实现：

1. CLI 单元测试先覆盖新目录扫描、旧路径失效、放大模型排除、显示名、唯一解析和后端限制。
2. Python 后端测试先覆盖四种 GIMM、三种 GMFSS 的架构识别、必要键验证、GMFSS Base 无 `rife` 加载分支和非法后端组合。
3. 转换脚本对每个输出检查文件大小、键集合、模型类型和可反序列化；转换前后检查 1 GB 停止条件。
4. 构建插件 DLL 和单文件 CLI EXE，验证文件版本为 1.4.2，且 EXE 内嵌最新 DLL。
5. 在实际便携目录运行 `--list-interp-models --json`、`--check` 和最小双帧推理测试。
6. 启动插件验证模型下拉框、自动 CUDA 切换、配置保存、下载分类与下载后刷新。
7. 最后更新根目录 `project.md`，把 1.4.2 设为默认有效版本并记录修改、验证结果和未决风险。

## 成功标准

- 所有现有 RIFE NCNN 模型从新目录被发现，旧硬编码不再存在。
- 四个 GIMM-VFI 和三个 GMFSS 上游推理版本转换为独立组合权重并可由后端识别。
- CLI、插件和下载器使用相同的相对路径与架构名称。
- 不支持的后端组合在任务启动前被阻止。
- DLL、EXE 和部署后端均来自 1.4.2 源码并通过新鲜验证。
- `project.md` 反映最终实际状态。
