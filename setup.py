from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "한국 지역 정책 검색 API 패키지"

# requirements.txt 읽기
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "sentence-transformers>=2.2.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "faiss-cpu>=1.7.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "requests>=2.31.0",
            "pydantic>=2.5.0",
            "PyYAML>=6.0"
        ]

setup(
    name="policy-chatbot-api",
    version="1.0.0",
    author="KDT Hackathon Team",
    author_email="team@example.com",
    description="한국 지역 정책 검색 API 패키지",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/LEEYH205/kdt_policy_chatbot_api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "policy-api=policy_chatbot_api.api_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "policy_chatbot_api": [
            "data/*.csv",
            "*.pkl",
            "*.json",
        ],
    },
    keywords="policy, api, korean, regional, government",
    project_urls={
        "Bug Reports": "https://github.com/LEEYH205/kdt_policy_chatbot_api/issues",
        "Source": "https://github.com/LEEYH205/kdt_policy_chatbot_api",
        "Documentation": "https://github.com/LEEYH205/kdt_policy_chatbot_api/blob/main/README.md",
    },
) 