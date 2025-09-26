from setuptools import setup, find_packages

setup(
    name='tokensmith',
    version='0.1.0',
    description='A package for managing datasets with editing, inspecting, sampling, exporting, and searching functionalities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aflah02/TokenSmith', 
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Apache 2.0 License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0', 
        'tqdm>=4.62.0',
    ],
    extras_require={
        'ui': [
            'streamlit>=1.20.0',
            'altair>=4.2.0',
        ],
        'search': [
            'tokengrams>=0.3.0',
        ],
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.0.0', 
            'mkdocstrings[python]>=0.24.0',
            'mkdocstrings-python>=1.7.0',
        ],
        'all': [
            'streamlit>=1.20.0',
            'altair>=4.2.0',
            'tokengrams>=0.3.0',
        ],
    },
)
