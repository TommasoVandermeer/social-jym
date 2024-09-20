from setuptools import setup, find_packages

setup(
    name='socialjym',
    version='0.0.1',
    packages= find_packages(),
    install_requires = [
    'jax>=0.4.30',
    'jax_tqdm>=0.2.2',
    'matplotlib>=3.9.1',
    'dm-haiku>=0.0.12',
    'optax>=0.2.3',
    'notebook>=7.2.2',
    ]
)