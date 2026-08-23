# VideoEnhancer 1.4.2 Frame Interpolation Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable VideoEnhancer 1.4.2 with the new `Frame-Interpolation` model layout and end-to-end CUDA/PyTorch support for all released GIMM-VFI and GMFSS Fortuna inference variants.

**Architecture:** Keep the existing unified RVE frame pipeline and package every upstream variant as one self-contained checkpoint. A small metadata inspector validates checkpoints and selects RIFE, GIMM-R, GIMM-F, or GMFSS before rendering; the CLI and plugin use paths relative to `models\Frame-Interpolation` as their shared model identity.

**Tech Stack:** .NET 10/C#, VB.NET WinForms, Python 3.12, PyTorch 2.9 CUDA 13, unittest, ModelScope-backed downloader

**Spec:** `1.4.2/docs/superpowers/specs/2026-08-23-frame-interpolation-models-design.md`

## Global Constraints

- Modify only `1.4.2` and `C:\PortableSoft\VideoEnhancer-CLI` backend/deployment files.
- Publish one combined checkpoint per upstream inference variant.
- Stop and ask before keeping any source or converted model file larger than 1 GB.
- GIMM-VFI and GMFSS are CUDA/PyTorch-only; never advertise NCNN, ONNX, or native TensorRT support for them.
- Preserve all unrelated VideoEnhancer behavior.
- The workspace has no Git metadata, so do not run commit steps.

---

### Task 1: Checkpoint Metadata and Validation

**Files:**
- Create: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\test_interpolation_models.py`
- Create: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolationModelInfo.py`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\rve-backend.py`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateTorch.py`

**Interfaces:**
- Produces: `inspect_interpolation_checkpoint(path: str) -> InterpolationModelInfo`
- Produces: `validate_interpolation_backend(info: InterpolationModelInfo, backend: str) -> None`
- Produces: `InterpolationModelInfo(architecture: str, variant: str, model_type: str | None)`

- [ ] **Step 1: Write failing metadata tests**

Add tests that create tiny checkpoints with `torch.save` and assert:

```python
info = inspect_interpolation_checkpoint(gimm_r_path)
self.assertEqual((info.architecture, info.variant), ("gimm", "r"))

info = inspect_interpolation_checkpoint(gmfss_base_path)
self.assertEqual((info.architecture, info.model_type), ("gmfss", "base"))

with self.assertRaisesRegex(ValueError, "仅支持 CUDA/PyTorch"):
    validate_interpolation_backend(info, "ncnn")
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `python\python\python.exe python\backend\test_interpolation_models.py -v`

Expected: assertion failure that `inspect_interpolation_checkpoint` is not yet available.

- [ ] **Step 3: Implement the inspector and backend guard**

Use metadata first and structural keys second. Required combined-checkpoint keys are:

```python
GIMM_REQUIRED = {
    "r": {"gimmvfi", "flow_estimator"},
    "f": {"gimmvfi", "flow_estimator"},
}
GMFSS_REQUIRED = {
    "base": {"flownet", "metricnet", "feat_ext", "fusionnet"},
    "union": {"rife", "flownet", "metricnet", "feat_ext", "fusionnet"},
}
```

`rve-backend.py` calls the guard before constructing `Render`. `InterpolateFactory` uses the returned architecture/variant and raises a descriptive `ValueError` for unknown content.

- [ ] **Step 4: Run focused and existing backend tests**

Run:

```powershell
python\python\python.exe python\backend\test_interpolation_models.py -v
python\python\python.exe python\backend\test_rve_backend.py -v
```

Expected: all tests pass.

---

### Task 2: Upstream Model Conversion

**Files:**
- Create: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\tools\convert_frame_interpolation_models.py`
- Create: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\test_convert_frame_interpolation_models.py`
- Create outputs under: `C:\PortableSoft\VideoEnhancer-CLI\models\Frame-Interpolation\GIMM-VFI`
- Create outputs under: `C:\PortableSoft\VideoEnhancer-CLI\models\Frame-Interpolation\GMFSS`

**Interfaces:**
- Produces: `convert_gimm(checkpoint, flow_estimator, variant, output) -> Path`
- Produces: `convert_gmfss(train_log, model_type, output) -> Path`
- Produces checkpoint metadata `{"architecture": ..., "variant": ..., "model_type": ..., "source": ...}`

- [ ] **Step 1: Write failing converter tests**

Tests use tiny synthetic state dicts and assert one-file output, exact metadata, required keys, reloadability, and rejection above `MAX_MODEL_BYTES = 1_073_741_824`.

- [ ] **Step 2: Run tests and verify converter functions are absent**

Run: `python\python\python.exe python\backend\test_convert_frame_interpolation_models.py -v`

Expected: failing assertions for absent converter functions.

- [ ] **Step 3: Implement the minimal converter**

For GIMM, normalize `checkpoint["state_dict"]` into `gimmvfi` and the RAFT/FlowFormer checkpoint into `flow_estimator`. For GMFSS, recursively find the exact upstream filenames `flownet.pkl`, `metric.pkl`, `feat.pkl`, `fusionnet.pkl`, and optional `rife.pkl`, then save normalized keys.

- [ ] **Step 4: Download upstream inference assets to a temporary directory**

Download the six GIMM repository files needed by four variants and the three GMFSS Model Zoo archives. Before retaining every response, verify `Content-Length <= 1_073_741_824`.

- [ ] **Step 5: Convert all seven variants**

Create exactly:

```text
GIMM-VFI/GIMM-VFI-R.pth
GIMM-VFI/GIMM-VFI-R-LPIPS.pth
GIMM-VFI/GIMM-VFI-F.pth
GIMM-VFI/GIMM-VFI-F-LPIPS.pth
GMFSS/GMFSS-Fortuna-Base.pth
GMFSS/GMFSS-Fortuna-Union.pth
GMFSS/GMFSS-Fortuna-Union-AnimeRun.pth
```

- [ ] **Step 6: Verify converted assets**

Run the converter `--verify` mode over `models\Frame-Interpolation`. Expected: seven valid CUDA/PyTorch combined checkpoints, each below 1 GB, plus the existing five RIFE NCNN directories.

---

### Task 3: GIMM-R, GIMM-F, and GMFSS Runtime Loaders

**Files:**
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateGIMM.py`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateGMFSS.py`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateArchs\GMFSS\GMFSS.py`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateArchs\DetectInterpolateArch.py`
- Create: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateArchs\GIMM\gimmvfi_f.py`
- Create/copy minimal runtime tree: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\src\pytorch\InterpolateArchs\GIMM\flowformer`
- Modify: `C:\PortableSoft\VideoEnhancer-CLI\python\backend\requirements.txt`
- Modify portable packages under: `C:\PortableSoft\VideoEnhancer-CLI\python\python\Lib\site-packages`

**Interfaces:**
- Consumes: normalized checkpoint format from Tasks 1-2.
- Produces: `InterpolateGIMMTorch` supporting variants `r` and `f` through the same frame generator API.
- Produces: `GMFSS(..., model_type="base" | "union")` without unconditional `rife` access.

- [ ] **Step 1: Add failing loader-construction tests**

Patch heavyweight tensor construction only at the GPU boundary and assert that GIMM chooses the R or F network class from metadata, GMFSS Base does not construct IFNet, and Union does.

- [ ] **Step 2: Verify the focused tests fail on current loaders**

Run: `python\python\python.exe python\backend\test_interpolation_models.py -v`

Expected: failures for GIMM-F and GMFSS Base.

- [ ] **Step 3: Import the minimal official GIMM-F runtime**

Copy only inference modules reachable from `gimmvfi_f.py` and FlowFormer. Preserve upstream license notices. Adapt relative imports to the existing backend package and use the existing torch soft-splat implementation instead of compiling `alt_cuda_corr`.

- [ ] **Step 4: Install portable runtime dependencies**

Install `scipy`, `timm`, `yacs`, and `loguru` into the bundled Python and add pinned compatible requirements. Verify all four imports using `python\python\python.exe`.

- [ ] **Step 5: Update GIMM and GMFSS loaders**

Load normalized keys `gimmvfi`/`flow_estimator`; choose GIMM-R or GIMM-F from metadata. In GMFSS, read `model_type` before accessing `rife`, instantiate IFNet only for Union, and retain the PyTorch soft-splat fallback when CuPy is unavailable.

- [ ] **Step 6: Run tests and checkpoint load smoke tests**

Run the two backend test files, then instantiate all seven models at a minimal padded resolution without rendering a video. Expected: every checkpoint loads on CUDA and no model silently falls back to another architecture.

---

### Task 4: CLI Scanning, Resolution, and Version 1.4.2

**Files:**
- Create: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\cli\tests\test_frame_interpolation_cli.py`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\cli\Program.cs`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\cli\VideoEnhancer.csproj`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\cli\README.md`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\cli\build.ps1`

**Interfaces:**
- Produces: `FrameInterpolationDir`, `RifeInterpolationDir`, `GmfssInterpolationDir`, `GimmInterpolationDir`.
- Produces: `--list-interp-models --json` relative names rooted at `Frame-Interpolation`.

- [ ] **Step 1: Write failing deployed-CLI tests**

Invoke `C:\PortableSoft\VideoEnhancer-CLI\videoenhancer.exe --list-interp-models --json` for NCNN and CUDA. Assert NCNN contains `RIFE/rife-v4.25`, CUDA contains all seven new checkpoint paths, and `--list-models --json -backend cuda` contains none of them.

- [ ] **Step 2: Run and verify the old CLI fails**

Run: `python 1.4.2\cli\tests\test_frame_interpolation_cli.py -v`

Expected: old `models\RIFE` scan returns no moved RIFE models.

- [ ] **Step 3: Implement the unified scan and resolver**

Replace `IsInRifeDirectory` with an exclusion for the full `Frame-Interpolation` root. NCNN discovers paired RIFE folders; CUDA discovers `.pth/.pt/.pkl` files under all three architecture folders. Display and resolve relative paths from `Frame-Interpolation`, with unique basename fallback for old saved RIFE values.

- [ ] **Step 4: Update all 1.4.2 CLI version surfaces**

Set `ToolVersion`, project `<Version>`, README paths/text, and build comments to `1.4.2`.

- [ ] **Step 5: Build and deploy the CLI**

Run `1.4.2\cli\build.ps1`, then copy the resulting single EXE to `C:\PortableSoft\VideoEnhancer-CLI\videoenhancer.exe`.

- [ ] **Step 6: Re-run CLI tests**

Expected: NCNN and CUDA lists match their architectures, and upscale lists exclude all interpolation assets.

---

### Task 5: Plugin UI and Downloader Integration

**Files:**
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\VideoEnhancerPlugin\PluginPanel.vb`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\VideoEnhancerPlugin\PluginConfig.vb`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\README.md`
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\1.4.2\deploy.ps1`

**Interfaces:**
- Consumes: architecture-prefixed values from `--list-interp-models --json`.
- Produces: automatic CUDA selection for `GMFSS/` and `GIMM-VFI/` entries.
- Produces: downloader categories and installed-state checks rooted at `Frame-Interpolation`.

- [ ] **Step 1: Update model selection behavior**

In `OnInterpModelSelected`, detect the two CUDA-only prefixes with ordinal-ignore-case comparison. Set `_config.InterpBackend = "cuda"`, synchronize `_cmbInterpBackend`, refresh the list once without event recursion, and show a concise status message.

- [ ] **Step 2: Update downloader classification and installed checks**

Recognize remote paths under `Frame-Interpolation/RIFE`, `/GMFSS`, and `/GIMM-VFI`; test exact local target existence. Replace all `models\RIFE` checks with `models\Frame-Interpolation\RIFE`.

- [ ] **Step 3: Update 1.4.2 plugin-facing text**

Replace stale RIFE-only paths and version 1.4.1 user-agent/documentation text. Keep `PluginConfig` property names unchanged for backward compatibility, but update their comments and accepted values.

- [ ] **Step 4: Build the plugin and then rebuild the embedding CLI**

Run `1.4.2\VideoEnhancerPlugin\build.ps1`, then `1.4.2\cli\build.ps1`, ensuring the EXE embeds the newly built DLL. Deploy both artifacts to their existing runtime targets.

- [ ] **Step 5: Verify visible behavior**

Open the plugin and confirm the dropdown shows architecture/version paths, GIMM/GMFSS auto-select CUDA, RIFE remains selectable with NCNN, saved selections reload, and downloader groups/installed states use the new layout.

---

### Task 6: Full Verification and Project Index

**Files:**
- Modify: `C:\Users\ARXChem\Documents\LakeUI-2\videoenhancer.3fui\project.md`

**Interfaces:**
- Consumes: all prior deliverables.
- Produces: current project status and reproducible verification evidence.

- [ ] **Step 1: Run the complete automated test suite**

Run backend tests, converter tests, ordered-backend tests, and CLI integration tests. Record exact totals and failures.

- [ ] **Step 2: Verify build artifacts and versions**

Check DLL/EXE timestamps, file hashes, CLI `--version`, and .NET file version. Verify the deployed EXE hash matches the 1.4.2 release EXE and that it contains the current plugin resource.

- [ ] **Step 3: Run minimal real inference**

Use a tiny two-frame/video fixture to render at least one output with RIFE NCNN, one GIMM-R, one GIMM-F, GMFSS Base, and GMFSS Union on CUDA. Verify output opens, frame count matches the requested factor, and each log names the selected architecture.

- [ ] **Step 4: Run environment and downloader checks**

Run deployed `--check`, both interpolation list commands, and the remote download list. If converted assets are absent from ModelScope, report that external upload remains unperformed rather than claiming remote download availability.

- [ ] **Step 5: Update `project.md`**

Set 1.4.2 as the default effective version; update the model layout, backend file index, scan/download flow, verification commands, dependency notes, and change record with actual results and remaining risks.

- [ ] **Step 6: Re-read the spec and verify every success criterion**

Report each criterion as passed or blocked with fresh command evidence. Do not claim completion when a required inference variant or remote downloader path remains unverified.
