from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="super-tictactoe-rl",
    version="0.1.0",
    author="Assignment2 Team",
    description="Super Tic-Tac-Toe Reinforcement Learning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/super-tictactoe-rl",
    package_dir={
        "": "src",
    },
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
