import json
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


DEPLOYED_EXE = Path(r"C:\PortableSoft\VideoEnhancer-CLI\videoenhancer.exe")
INTERPOLATION_ROOT = Path(r"C:\PortableSoft\VideoEnhancer-CLI\models\Frame-Interpolation")


def run_json_command(*arguments):
    result = subprocess.run(
        [str(DEPLOYED_EXE), *arguments, "--json"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise AssertionError(
            f"CLI exited with {result.returncode}: {result.stderr.strip()}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


class TestFrameInterpolationCli(unittest.TestCase):
    def test_recursive_scan_uses_format_not_second_level_folder_name(self):
        unique = "scan-test-" + uuid.uuid4().hex
        test_root = INTERPOLATION_ROOT / unique
        ncnn_dir = test_root / "arbitrary-family" / "nested-ncnn"
        cuda_file = test_root / "another-family" / "nested-cuda.pth"
        engine_file = test_root / "future-family" / "nested-tensorrt.engine"
        try:
            ncnn_dir.mkdir(parents=True)
            (ncnn_dir / "model.param").write_bytes(b"test")
            (ncnn_dir / "model.bin").write_bytes(b"test")
            cuda_file.parent.mkdir(parents=True)
            cuda_file.write_bytes(b"test")
            engine_file.parent.mkdir(parents=True)
            engine_file.write_bytes(b"test")

            ncnn_models = run_json_command(
                "--list-interp-models", "-interp-backend", "ncnn"
            )
            cuda_models = run_json_command(
                "--list-interp-models", "-interp-backend", "cuda"
            )
            tensorrt_models = run_json_command(
                "--list-interp-models", "-interp-backend", "tensorrt"
            )
            upscale_tensorrt_models = run_json_command(
                "--list-models", "-backend", "tensorrt"
            )

            self.assertIn(
                f"{unique}/arbitrary-family/nested-ncnn", ncnn_models
            )
            self.assertIn(
                f"{unique}/another-family/nested-cuda", cuda_models
            )
            self.assertIn(
                f"{unique}/future-family/nested-tensorrt", tensorrt_models
            )
            self.assertNotIn(
                f"Frame-Interpolation/{unique}/future-family/nested-tensorrt",
                upscale_tensorrt_models,
            )
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_cuda_lists_all_gimm_and_gmfss_variants(self):
        models = run_json_command("--list-interp-models", "-interp-backend", "cuda")

        expected = {
            "GIMM-VFI/GIMM-VFI-R",
            "GIMM-VFI/GIMM-VFI-R-LPIPS",
            "GIMM-VFI/GIMM-VFI-F",
            "GIMM-VFI/GIMM-VFI-F-LPIPS",
            "GMFSS/GMFSS-Fortuna-Base",
            "GMFSS/GMFSS-Fortuna-Union",
            "GMFSS/GMFSS-Fortuna-Union-AnimeRun",
        }
        self.assertTrue(expected.issubset(set(models)))

    def test_cuda_upscale_list_excludes_frame_interpolation_models(self):
        models = run_json_command("--list-models", "-backend", "cuda")

        self.assertFalse(
            any(
                model.startswith(("RIFE/", "GIMM-VFI/", "GMFSS/"))
                for model in models
            )
        )


if __name__ == "__main__":
    unittest.main()
