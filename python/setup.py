from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


extra_compile_args: list[str] = []
if sys.platform != "win32":
    extra_compile_args.extend(["-O3", "-std=c++17"])
else:
    extra_compile_args.extend(["/O2", "/std:c++17"])

build_accel = os.environ.get("CCGEN_BUILD_ACCEL", "1").lower() not in {
    "0",
    "false",
    "no",
}

ext_modules = []
if build_accel:
    ext_modules.append(
        Extension(
            "ccgen._wickaccel",
            sources=["ccgen/_wickaccel.cpp"],
            language="c++",
            extra_compile_args=extra_compile_args,
            optional=True,
        )
    )


setup(
    ext_modules=ext_modules,
)
