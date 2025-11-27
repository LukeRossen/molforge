from setuptools import setup, find_packages

setup(
    name="molforge",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',      # Data manipulation
        'numpy>=1.21.0',      # Numerical operations (explicit dependency)
        'rdkit',              # Cheminformatics toolkit
        'requests>=2.25.0',   # HTTP requests (ChEMBL database downloader)
        'tqdm>=4.60.0',       # Progress bars (database downloads)
        'psutil>=5.8.0',      # System utilities (multiprocessing, memory)
        'joblib>=1.0.0',      # Parallel processing (RDKit backend uses cloudpickle)
    ],
    extras_require={
        'molbox': ['molbox'],                    # Molbox integration
        'cli': [
            'pyyaml>=5.4',                      # YAML config file support
            'questionary>=2.0.0',               # Interactive CLI prompts
        ],
        'openeye': ['openeye-toolkits'],        # OpenEye OMEGA backend (optional)
        'dev': [                                # Development tools
            'pytest>=7.0.0',
            'pytest-mock>=3.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'molforge=molforge.cli.main:main',
        ],
    },
    python_requires='>=3.8',
)