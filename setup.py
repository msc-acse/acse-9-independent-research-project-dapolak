import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slugdetection",
    version="1.0.0,
    author="dapolak",
    author_email="deirdree.polak@gmail.com",
    description="Slugging Detection package using feature engineering "
                "and supervised classification for offshore oil wells.",
    long_description=long_description,
    url="https://github.com/msc-acse/acse-9-project-plan-dapolak",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independant",
    ],
)
