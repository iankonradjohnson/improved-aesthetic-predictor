from setuptools import setup, find_packages

setup(
    name="aesthetic-predictor",
    version="0.1.0",
    description="A machine learning model to predict the aesthetic quality of images",
    author="Ian Johnson",
    author_email="example@example.com",
    packages=find_packages(include=[""]),
    py_modules=["aesthetic_predictor"],
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
    ],
    dependency_links=[
        "git+https://github.com/openai/CLIP.git#egg=clip"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "": ["*.pth"],
    },
)