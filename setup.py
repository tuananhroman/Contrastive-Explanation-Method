import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DAI-Labor-CEM-tuananhroman",
    version="0.0.1",
    author="Tuan Anh Le",
    author_email="tuananh.le@dai-labor.de",
    description="[DAI Labor] CEM integration into DEMO app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuananhroman/Contrastive-Explanation-Method",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)