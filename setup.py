import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="aml_end_to_end",
    version="0.1.0",
    description='End to end pipeline for anti money laundering',
    author="Hopsworks AB",
    author_email="davit@logicalclocks.com",
    license="Apache License 2.0",
    keywords="Hopsworks, Feature Store, Spark, Machine Learning, MLOps, DataOps, Fraud, Money laundering",
    url="https://github.com/logicalclocks/AMLend2end.git",
    download_url="https://github.com/logicalclocks/AMLend2end/releases/tag/0.1.0",
    packages=find_packages(),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
)