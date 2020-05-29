from setuptools import setup, find_packages
import marynlp

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="marynlp",
    version=marynlp.__version__,
    description="A NLP Approach towards the Swahili Language",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Kevin James",
    author_email="kevin.al.james@gmail.com",
    url="https://github.com/inspiredideas/marynlp",
    packages=find_packages(exclude="_tests"),  # same as name
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6",
)