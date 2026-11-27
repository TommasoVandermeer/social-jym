import sys
from setuptools import setup, find_packages

if sys.version_info[:2] not in [(3, 10), (3, 13)]:
    error_msg = (
        "\n" + "="*60 + "\n"
        "INSTALLATION ERROR\n"
        "The package 'socialjym' supports EXCLUSIVELY:\n"
        "  - Python 3.10\n"
        "  - Python 3.13\n"
        f"You are trying to install using Python {sys.version_info.major}.{sys.version_info.minor}\n"
        "Installation aborted."
        "\n" + "="*60 + "\n"
    )
    sys.exit(error_msg)

setup(
    name='socialjym',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # Python 3.10 (legacy: recent JAX requires Python >= 3.12; kept for the Docker image)
        'jax==0.4.30; python_version == "3.10"',
        'jaxlib==0.4.30; python_version == "3.10"',
        'scipy==1.14.0; python_version == "3.10"',
        'scikit-learn==1.7.2; python_version == "3.10"',
        'flax==0.10.2; python_version == "3.10"',
        'optax==0.2.4; python_version == "3.10"',
        # Python 3.13
        'jax==0.12.1; python_version == "3.13"',
        'jaxlib==0.12.1; python_version == "3.13"',
        'scipy==1.18.1; python_version == "3.13"',
        'scikit-learn==1.9.0; python_version == "3.13"',
        'flax==0.12.12; python_version == "3.13"',
        'optax==0.2.9; python_version == "3.13"',
        # Common dependencies
        'jax_tqdm==0.2.2',
        'matplotlib==3.9.1',
        'notebook==7.2.2',
        'pandas==2.2.3',
        
    ],
    # On Python 3.10 (jax 0.4.30) only CUDA 12 is available, and GPUs newer than Hopper
    # (e.g. RTX 50xx) need docker/ptxas_wrapper.sh.
    extras_require={
        'cuda12': [
            'jax[cuda12]==0.4.30; python_version == "3.10"',
            'jax[cuda12]==0.12.1; python_version == "3.13"',
        ],
        'cuda13': [
            'jax[cuda13]==0.12.1; python_version == "3.13"',
        ],
    }
)