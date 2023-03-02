import os
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf8')
homepage = "https://github.com/FeliMe/RoDeO"

setup(
    name="rodeo",
    version="1.0",
    description="Robust Detection Outcome: A Metric for Pathology Detection in Medical Images",
    author="Felix Meissen, Philip Müller",
    author_email="felix.meissen@tum.de, philip.j.mueller@tum.de",
    url=homepage,
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    packages=["rodeo"],
    include_package_data=True,
    install_requires=["scipy", "torch", "torchmetrics", "torchvision"],
    keywords=["deep learning", "machine learning", "pytorch", "metrics", "AI"],
    project_urls={
        "Bug Tracker": os.path.join(homepage, "issues"),
        "Source Code": homepage,
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ],
)
