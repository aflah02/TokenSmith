from setuptools import setup, find_packages

setup(
    name="tokensmith",
    version="0.1.1",
    description=(
        "A Package for Streamlining Data Editing, Search, and Inspection for "
        "Large-Scale Language Model Training and Interpretability"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mohammad Aflah Khan, Ameya Godbole",
    author_email="afkhan@mpi-sws.org, ameyagod@usc.edu",
    license="MIT",
    url="https://github.com/aflah02/tokensmith",
    project_urls={
        "Homepage": "https://github.com/aflah02/tokensmith",
        "Repository": "https://github.com/aflah02/tokensmith",
    },
    keywords=[
        "dataset",
        "management",
        "editing",
        "sampling",
        "exporting",
        "searching",
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "tqdm>=4.67.1",
    ],
    extras_require={
        "docs": [
            "mkdocs>=1.6.1",
            "mkdocs-material>=9.6.14",
            "mkdocstrings[python]>=0.29.1",
            "mkdocstrings-python>=1.16.12",
            "mkdocs-autorefs>=1.4.2",
            "mkdocs-material-extensions>=1.3.1",
            "mkdocs-get-deps>=0.2.0",
            "mkdocs-jupyter>=0.25.1",
        ],
        "ui": [
            "streamlit>=1.46.0",
            "altair>=5.5.0",
        ],
        "search": [
            "tokengrams>=0.3.3",
        ],
        "all": [
            "streamlit>=1.46.0",
            "altair>=5.5.0",
            "tokengrams>=0.3.3",
        ],
    },
    include_package_data=True,
)