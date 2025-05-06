#!/usr/bin/env python3

import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.dir_util import remove_tree

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

with open("README.md", "r", encoding="utf-8") as fp:
    long_description = fp.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        print("building extension...")

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"
        # cfg = "Debug"

        conda_prefix = os.environ["CONDA_PREFIX"]

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            "-DGTEST=OFF",
            "-DDOCS=ON",
            f"-DGTEST_INCLUDE_DIRS={conda_prefix}/include/",
            f"-DGTEST_LIBRARIES={conda_prefix}/lib/libgtest.so",
            f"-DEIGEN3_INCLUDE_DIR={conda_prefix}/include/eigen3/",
            f"-Dpybind11_DIR={conda_prefix}/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/",
            "-DPYBIND11_FINDPYTHON=ON",
        ]
        build_args = ["--target", ext.name]

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator:
                cmake_args += ["-GNinja"]
            if self.compiler.compiler_type == 'darwin':
                cmake_args += [
                    "-DCMAKE_CXX_FLAGS=-fexperimental-library",
                    "-DCMAKE_MACOSX_DEPLOYMENT_TARGET=14_0"
                    ]

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                print(f"building in parallel with {self.parallel} threads")
                # CMake 3.12+ only.
                build_args += ["-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


# # # Clean old build/ directory if it exists
try:
    remove_tree("./build")
    print("Removed old build directory.")
except FileNotFoundError:
    print("No existing build directory found - skipping.")

setup(
    name="pybrush",
    version="0.0.1",  # TODO: use versionstr here
    author="William La Cava, Joseph D. Romano, Guilherme Aldeia",
    author_email='williamlacava@gmail.com', 
    license="GNU General Public License v3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lacava/brush",
    project_urls={
        "Bug Tracker": "https://github.com/lacava/brush/issues",
    },
    package_dir={"": "."},
    packages=find_packages(where="."),
    # cmake_install_dir="src/",
    python_requires=">=3.6",
    install_requires=["numpy", "scikit-learn", "sphinx"],
    tests_require=["pytest", "pmlb"],
    extras_require={"docs": ["sphinx_rtd_theme", "maisie_sphinx_theme", "breathe"]},
    ext_modules=[CMakeExtension("_brush")],
    cmdclass={"build_ext": CMakeBuild},
    test_suite="tests/python",
    zip_safe=False,
    include_package_data=True,
)
