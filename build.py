#!/usr/bin/env python3
"""
LingQi TanTong C Build Script
Target: SpacemiT K1 Muse Pi Pro (riscv64-linux-gnu)
"""

import os
import sys
import subprocess
import glob
import platform
import shutil

RISCV64_TOOLS = [
    "riscv64-linux-gnu-gcc",
    "riscv64-linux-gnu-g++",
    "riscv64-unknown-linux-gnu-gcc",
    "riscv64-buildroot-linux-gnu-gcc",
]

RISCV64_FLAGS = [
    "-march=rv64gcv",
    "-mabi=lp64d",
    "-O3",
    "-flto",
    "-ffast-math",
    "-fomit-frame-pointer",
    "-fPIC",
    "-Wall",
    "-Wextra",
]

RISCV64_LDFLAGS = [
    "-latomic",
    "-lpthread",
    "-lm",
    "-ldl",
]

HOST_COMPILERS = ["gcc", "clang", "cc"]


def find_riscv_compiler():
    for tool in RISCV64_TOOLS:
        if shutil.which(tool):
            return tool
    return None


def find_host_compiler():
    for compiler in HOST_COMPILERS:
        if shutil.which(compiler):
            return compiler
    return None


def get_source_files():
    src_dir = "src"
    sources = []
    for pattern in ["*.c"]:
        sources.extend(glob.glob(os.path.join(src_dir, pattern)))
    return sorted(sources)


def build_for_riscv64():
    compiler = find_riscv_compiler()
    if not compiler:
        print("ERROR: No RISC-V cross compiler found!")
        print("Install: sudo apt install gcc-riscv64-linux-gnu")
        print("Or set up SpacemiT Bianbu SDK toolchain")
        return False

    print(f"RISC-V Compiler: {compiler}")
    return do_build(compiler, RISCV64_FLAGS, RISCV64_LDFLAGS, "riscv64")


def build_for_host():
    compiler = find_host_compiler()
    if not compiler:
        print("ERROR: No C compiler found!")
        return False

    print(f"Host Compiler: {compiler}")
    print("WARNING: Building for host (dev/testing only)")
    print("Target platform is SpacemiT K1 Muse Pi Pro (riscv64)")

    cflags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Iinclude"]
    ldflags = ["-lm", "-lpthread"]
    return do_build(compiler, cflags, ldflags, "host")


def do_build(compiler, cflags, ldflags, target):
    sources = get_source_files()
    main_src = "src/main.c"
    test_src = "tests/test_basic.c"

    if main_src in sources:
        sources.remove(main_src)

    cflags_all = ["-std=c11"] + cflags + ["-Iinclude", "-Isrc"]

    objects = []
    for src in sources:
        obj = src.replace(".c", f".{target}.o")
        objects.append(obj)
        cmd = [compiler] + cflags_all + ["-c", src, "-o", obj]
        print(f"  CC {src}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    main_obj = main_src.replace(".c", f".{target}.o")
    cmd = [compiler] + cflags_all + ["-c", main_src, "-o", main_obj]
    print(f"  CC {main_src}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    binary = f"lingqi_tantong_{target}"
    cmd = [compiler] + cflags_all + objects + [main_obj, "-o", binary] + ldflags
    print(f"  LD {binary}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR linking: {result.stderr}")
        return False

    print(f"  Built: {binary}")

    test_obj = test_src.replace(".c", f".{target}.o")
    cmd = [compiler] + cflags_all + ["-c", test_src, "-o", test_obj]
    print(f"  CC {test_src}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    test_bin = f"test_basic_{target}"
    cmd = [compiler] + cflags_all + objects + [test_obj, "-o", test_bin] + ldflags
    print(f"  LD {test_bin}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR linking: {result.stderr}")
        return False

    print(f"  Built: {test_bin}")
    return True


def run_tests_riscv64():
    test_bin = "test_basic_riscv64"
    if not os.path.exists(test_bin):
        print("Test binary not found, building first...")
        if not build_for_riscv64():
            return False

    qemu = shutil.which("qemu-riscv64")
    if qemu:
        print(f"\nRunning tests via QEMU: {qemu}")
        try:
            result = subprocess.run([qemu, "-L", "/usr/riscv64-linux-gnu", f"./{test_bin}"],
                                   capture_output=False, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"QEMU test failed: {e}")
            return False
    else:
        print("QEMU not found. Copy test_basic_riscv64 to K1 Muse Pi Pro and run it.")
        print(f"  scp {test_bin} k1@<ip>:/home/k1/")
        print(f"  ssh k1@<ip> './{test_bin}'")
        return True


def clean():
    patterns = [
        "src/*.riscv64.o", "src/*.host.o",
        "tests/*.riscv64.o", "tests/*.host.o",
        "lingqi_tantong_riscv64", "lingqi_tantong_host",
        "test_basic_riscv64", "test_basic_host",
        "src/*.o", "tests/*.o",
        "lingqi_tantong", "test_basic",
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"  RM {f}")
            except OSError:
                pass

    dirs_to_clean = ["build"]
    for d in dirs_to_clean:
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                print(f"  RM {d}/")
            except OSError:
                pass

    return True


def deploy_to_k1(host, user="k1", path="/home/k1/lingqi_tantong"):
    binary = "lingqi_tantong_riscv64"
    if not os.path.exists(binary):
        print("Binary not found. Build first: python build.py riscv64")
        return False

    target = host if "@" in host else f"{user}@{host}"

    print(f"Deploying to {target}:{path}")
    try:
        subprocess.run(["ssh", target, f"mkdir -p {path}"], check=True)
        subprocess.run(["scp", binary, f"configs/default.yaml", f"{target}:{path}/"], check=True)
        subprocess.run(["scp", "-r", "models/", f"{target}:{path}/"], check=True)
        print(f"Deployed. Run on K1:")
        print(f"  ssh {target}")
        print(f"  cd {path} && ./lingqi_tantong_riscv64 --video_path test.mp4")
    except subprocess.CalledProcessError as e:
        print(f"Deploy failed: {e}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="LingQi TanTong C Build Script for SpacemiT K1 Muse Pi Pro"
    )
    parser.add_argument(
        "command",
        choices=["riscv64", "host", "build", "test", "clean", "deploy"],
        default="riscv64",
        nargs="?",
        help="Build target: riscv64 (K1), host (dev), build (legacy), test, clean, deploy",
    )
    parser.add_argument(
        "--deploy-host",
        help="K1 IP or user@host for deploy command",
        default="192.168.1.100",
    )
    parser.add_argument(
        "--deploy-user",
        help="SSH user for deploy",
        default="k1",
    )
    args = parser.parse_args()

    if args.command == "clean":
        return clean()
    elif args.command == "riscv64":
        return build_for_riscv64()
    elif args.command == "host" or args.command == "build":
        return build_for_host()
    elif args.command == "test":
        return run_tests_riscv64()
    elif args.command == "deploy":
        if not build_for_riscv64():
            return False
        return deploy_to_k1(args.deploy_host, args.deploy_user)

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)