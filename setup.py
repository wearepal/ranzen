from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="palkit",
    version="0.1.1",
    author="PAL",
    author_email="info@predictive-analytics-lab.com",
    description="Useful functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/predictive-analytics-lab/palkit",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests.*", "tests"]),
    package_data={"kit": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=[
        "hydra-core == 1.1.0.dev3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    keywords=["typing", "python"],
)
