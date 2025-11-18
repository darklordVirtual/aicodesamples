from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-integrasjoner-norsk",
    version="1.0.0",
    author="Stian Skogbrott",
    author_email="kontakt@luftfiber.no",
    description="Kodeeksempler for boken 'AI og Integrasjoner: Fra Grunnleggende til Avansert'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luftfiber/ai-integrasjoner-norsk",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.54.0",
        "chromadb>=0.5.0",
        "mcp>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "PyPDF2>=3.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
)
