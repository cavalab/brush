#!/usr/bin/env python3

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
    )
from setuptools import find_packages

with open("README.md", 'r', encoding="utf-8") as fp:
    long_description = fp.read()

setup(
    name="brush-gp",
    version="0.0.1",
    author="William La Cava and Joseph D. Romano",
    author_email="joseph.romano@pennmedicine.upenn.edu",  # can change to Bill
    license="GNU General Public License v3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lacava/brush",
    project_urls={
        "Bug Tracker": "https://github.com/lacava/brush/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    cmake_install_dir="src/brush",
    python_requires=">=3.6",
)