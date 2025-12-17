from setuptools import find_packages, setup

setup(
    name="ai_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
    ],
)
