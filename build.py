import setuptools
from setuptools import find_packages


setuptools.setup(    
    name='CLEAN',
    version='0.1',
    description='CLEAN: Enzyme Function Prediction using Contrastive Learning',
    author='Tianhao Yu, Haiyang Cui, Jianan Canal Li, Yunan Luo, Huimin Zhao',
    url='https://github.com/tttianhao/CLEAN',
    project_urls={
        "Bug Tracker": "https://github.com/tttianhao/CLEAN",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
)