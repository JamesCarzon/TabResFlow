from setuptools import setup, find_packages

setup(
    name="tabresflow",
    version="0.1.0",
    description="TabResFlow - probabilistic regression submodule",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.8",
)