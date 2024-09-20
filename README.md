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
Install the submodules.
```
cd social-jym/JHSFM
python3 install setup.py
cd ../JSFM
python3 install setup.py
cd ..
```
Install the main package.
```
python3 install setup.py
```