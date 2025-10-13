# social-jym
An environment based on JAX to train mobile robots within crowded environments. Includes several human motion models, several RL algorithms for social navigation and implements fast training and computing thanks to JAX.

## Installation
Create a virtual environment.
```
virtualenv socialjym
```
Activate the virtual environment.
```
source socialjym/bin/activate
```
Clone the repository and its submodules.
```
git clone --recurse-submodules https://github.com/TommasoVandermeer/social-jym.git
```
Install the submodules and the main package.
```
pip install -e social-jym/JHSFM
pip install -e social-jym/JSFM
pip install -e social-jym/JORCA
pip install -e social-jym
```