from setuptools import setup, find_packages
import marynlp

setup(
    name="marynlp",
    version=marynlp.__version__,
    description="A Swahili-first tool for NLP",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Kevin James",
    author_email="kevin@inspiredideas.io",
    license = "Apache-2.0",
    url="https://github.com/inspiredideas/marynlp",
    packages=find_packages(exclude=["tests",]),  # same as name
    keywords = "nlp swahili marynlp morphpiece sed",
    include_package_data=True,
    python_requires="==3.7.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
    ],
)