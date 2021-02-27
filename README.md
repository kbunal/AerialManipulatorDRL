# AerialManipulatorDRL
A comparative study between deep reinforcement learning approaches and conventional controller for aerial manipulation pick and place task

-Authors: Kazım Burak Ünal, Sinan Kalkan, Afşar Saranlı
-Paper: -

## Dependencies
-stablebaselines
-Box2D

## Installation

### Step 0
Create a directory for the project:

mkdir ~/AerialManipulatorDRL
cd ~/AerialManipulatorDRL

Instead of  ~/AerialManipulatorDRL any directory can be used. It is given just as an example.

### Step 1

Clone this repo
cd ~/AerialManipulatorDRL
git clone https://github.com/kbunal/AerialManipulatorDRL.git
cd AerialManipulatorDRL/

### Step 2
Install dependencies

python3 -m pip install Box2D
pip install stable-baselines[mpi]
pip install tensorflow==1.8.0

### Step 3
Setup the project 
cd ~/AerialManipulatorDRL
cd AerialManipulatorDRL/
pip install -e .

