#!/usr/bin/env python3

import setuptools

with open("README.md", 'r', encoding="utf-8") as fp:
    long_description = fp.read()

setuptools.setup(
    name="brush-gp",
    version="0.0.1",
    author="William La Cava and Joseph D. Romano",
    author_email="joseph.romano@pennmedicine.upenn.edu",  # can change to Bill
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lacava/brush",
    project_urls={
        "Bug Tracker": "https://github.com/lacava/brush/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)