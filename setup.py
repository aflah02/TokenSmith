from setuptools import setup, find_packages

setup(
    name='tokensmith',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for managing datasets with editing, inspecting, sampling, exporting, and searching functionalities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/tokensmith',  # Replace with your actual repository URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your actual license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
    ],
)