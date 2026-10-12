#!/usr/bin/env python
"""Extract matching NVIDIA OpenGL userspace libraries without replacing drivers.

Linux/Colab only. Prints shell exports for subsequent train/eval commands.
Does not install packages, alter the kernel driver, or configure Windows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def select_gl_package(driver: str, package_listing: str) -> str:
    for line in package_listing.splitlines():
        fields = line.split("|")
        if len(fields) >= 2:
            version = fields[1].strip()
            if version.split(":")[-1].startswith(driver + "-"):
                return version
    raise RuntimeError(f"No exact OpenGL package for NVIDIA driver {driver}; keep Xvfb/software rendering. Do not install another driver version.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/datn_nvidia_gl")
    args = parser.parse_args()
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    package = "libnvidia-gl-" + driver.split(".")[0]
    listing = subprocess.check_output(["apt-cache", "madison", package], text=True)
    version = select_gl_package(driver, listing)
    directory = Path(args.output).resolve() / driver
    directory.mkdir(parents=True, exist_ok=True)
    extracted = directory / "extracted"
    vendor = extracted / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not vendor.exists():
        subprocess.run(["apt-get", "download", f"{package}={version}"], cwd=directory, check=True)
        archives = list(directory.glob(f"{package}_*.deb"))
        if len(archives) != 1:
            raise RuntimeError(f"Expected exactly one downloaded package in {directory}")
        subprocess.run(["dpkg-deb", "-x", str(archives[0]), str(extracted)], check=True)
    library = extracted / "usr/lib/x86_64-linux-gnu"
    if not vendor.is_file() or not (library / "libEGL_nvidia.so.0").exists():
        raise RuntimeError("NVIDIA EGL package layout is unsupported; use Xvfb fallback")
    print("# Copy these exports into the SAME shell/cell as train/eval:")
    print(f'export LD_LIBRARY_PATH={shlex.quote(str(library))}:"${{LD_LIBRARY_PATH:-}}"')
    print(f"export __EGL_VENDOR_LIBRARY_FILENAMES={shlex.quote(str(vendor))}")
    print("export DATN_RENDER_BACKEND=egl")


if __name__ == "__main__":
    main()
