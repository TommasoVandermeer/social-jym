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
        # Python 3.10
        'jax==0.4.30; python_version == "3.10"',
        'jaxlib==0.4.30; python_version == "3.10"',
        'scipy==1.14.0; python_version == "3.10"',
        'scikit-learn==1.7.2; python_version == "3.10"',
        # Python 3.13
        'jax==0.4.34; python_version == "3.13"',
        'jaxlib==0.4.34; python_version == "3.13"',
        'scipy==1.14.1; python_version == "3.13"',
        'scikit-learn==1.9.0; python_version == "3.13"',
        # Common dependencies
        'jax_tqdm==0.2.2',
        'matplotlib==3.9.1',
        'dm-haiku==0.0.12',
        'optax==0.2.4',
        'notebook==7.2.2',
        'pandas==2.2.3',
        
    ],
    extras_require={
        # CUDA 13 requires a newer JAX line than this repository currently
        # supports. Keeping an incompatible extra here allowed pip to resolve a
        # materially different runtime, so CUDA 12 is the newest supported extra.
        'cuda12': [
            'jax[cuda12]==0.4.30; python_version == "3.10"',
            'jax[cuda12]==0.4.34; python_version == "3.13"',
        ],
        'cuda11': [
            'jax[cuda11]==0.4.30; python_version == "3.10"',
            'jax[cuda11]==0.4.34; python_version == "3.13"',
        ]
    }
)
