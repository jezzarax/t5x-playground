import os
import sys
import setuptools

setuptools.setup(
    name="t5x-playground",
    version="0.0.1",
    description="Base set of t5x tasks for a simple model",
    long_description="Base set of t5x tasks for a simple model",
    long_description_content_type="text/markdown",
    author="Alexey Kuntsevich",
    author_email="forspam@nowhere.com",
    url="http://github.com/jezzarax/t5x-playground",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=[],
    install_requires=[],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="sequence preprocessing nlp machinelearning",
)
