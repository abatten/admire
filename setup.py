from setuptools import setup, find_packages
from codecs import open

with open("README.rst", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name='admire',
    version='0.0.1',
    author='Adam Batten',
    author_email='adamjbatten@gmail.com',
    url='https://github.com/abatten/admire',
    project_urls={
        'Source Code': "https://github.com/abatten/admire"
        },
    description='Combine Electron Column Density Maps',
    long_description=long_description,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        ],
    package_dir={"admire": "admire"},
    packages=find_packages(),
    keywords=("astronomy astrophysics"),
    )
