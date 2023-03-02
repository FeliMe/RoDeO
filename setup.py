import os
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf8')
homepage = "https://github.com/FeliMe/RoDeO"

setup(
    name="rodeometric",
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
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn>=1.0.2"
    ],
    keywords=["deep learning", "machine learning", "metrics", "AI"],
    project_urls={
        "Bug Tracker": os.path.join(homepage, "issues"),
        "Source Code": homepage,
    },
    license="MIT",
    classifiers=[
        "Natural Language :: English",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
