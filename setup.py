from setuptools import setup, find_packages

setup(
    name="LLM_SCEs",            
    version="0.1",
    packages=find_packages(),             
    install_requires=[
        "python-Levenshtein==0.26.1",
        "regex==2024.11.6",
        "numpy==2.2.3",
        "torch==2.5.1+cu121",
        "transformers==4.49.0",
        "datasets==3.3.1",
        "matplotlib==3.10.0",
        "accelerate==1.4.0",
        "scipy==1.14.1",
        "seaborn==0.13.2",
    ],
    author="Zahra Dehghanighobadi",
    author_email="Zahra.Dehghanighobadi@ruhr-uni-bochum.de",
    description="In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs.",
    url="https://gitlab.ruhr-uni-bochum.de/ai-and-society/sces_llms.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.2",
)