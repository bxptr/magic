from setuptools import setup, find_packages

with open("requirements.txt") as handler:
    requirements = handler.readlines()

setup(
    name = "magic",
    author = "Aarush Gupta",
    description = "",
    packages = find_packages(),
    install_requires = requirements
)
