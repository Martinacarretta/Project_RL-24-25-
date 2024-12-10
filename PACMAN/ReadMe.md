## PROJECT - PACMAN
This repository contains the files used to train agent to play [Pacman](https://ale.farama.org/environments/pacman/)


## The game:
Pac-Man is a classic video game where players control a yellow, circular character navigating a maze
to eat pellets while avoiding ghosts. The game challenges players to strategize, collect bonus items,
and clear levels by consuming all pellets while using power pellets to temporarily defeat ghosts.
- Small pellets (yellow): +1
- Power pellets (pink): +5
- After consuming a power pellet, Pac-Man can eat ghosts (while they are blue). 
The first ghost provides a reward of +20, and the reward doubles for each additional ghost eaten.

## Action space:
Pacman’s action space is Discrete(5). The different actions correspond to:
- 0: NOOP
- 1: UP
- 2: RIGHT
- 3: LEFT
- 4: DOWN

## Observation space:
Observation space is Box(0, 255, (250, 160, 3), uint8). An array with shape (210, 160, 3) where each
element is an integer between 0 and 255

## Untrained agent:
First and foremost, we wanted to see how an untrained agent would perform to have an ide on how the game worked. The untrained agent typically gets a score within the range of 10 to 30. Occasionally,
it exceeds this range, achieving rewards between 50 and 100.
The gifs below represents an untrained agent. 

FALA POSAR GIFS (3) DEL UNTRAINED PACMAN

FALTA POSAR EL PLOT DELS REWARDS DEL TEST DE 10 EPISODIS UNTRAINED

## Training:
We decided to tackle this part of the project with 2 different algorithms:
- A2C
- PPO

## Training - A2C:
This part of the training has FALTA sections:

### Requirements:
The code requires the following libraries:
FALTA

to install them all, you can use:
FALTA

### Preprocessing:

FALTA

### Training:

FALTA parlar una mica del process que hem seguit

The training process is as follows:

FALTA

### Results:
The model is under the name "FALTA .pth"
FALTA GIFS DEL TEST DEL DQN I EL PLOT DELS REWARDS

### Conclusion:
FALTA

## Section 3 - Pong:
### The game:
You control the right paddle, you compete against the left paddle controlled by the computer. You each try to keep deflecting the ball away from your goal and into your opponent’s goal.
FALTA MIRAR SI ES AQUESTA VERSIO DE PONG O LA DE GYM

### Action space:
Pong's action space is Discrete(6). The different actions correspond to:
- 0: NOOP
- 1: FIRE
- 2: RIGHT
- 3: LEFT
- 4: RIGHTFIRE
- 5: LEFTFIRE

### Observation space:
Atari environments have three possible observation types: "rgb", "grayscale" and "ram".

- obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)
- obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)
- obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8), a grayscale version of the “rgb” type


### Untrained agent:
First and foremost, we wanted to see how an untrained agent would perform to have an ide on how the game worked.
The gifs below represents an untrained agent. 

FALA POSAR GIFS (3) DEL UNTRAINED PONG

FALTA POSAR EL PLOT DELS REWARDS DEL TEST DE 10 EPISODIS UNTRAINED

### Training:
We decided to tackle this part of the project with:
- FALTA

### Training - FALTA:
This part of the training has FALTA sections:

### Requirements:
The code requires the following libraries:
FALTA

to install them all, you can use:
FALTA

### Preprocessing:

FALTA

### Training:

FALTA parlar una mica del process que hem seguit

The training process is as follows:

FALTA

### Results:
The model is under the name "FALTA .pth"
FALTA GIFS DEL TEST DEL DQN I EL PLOT DELS REWARDS

### Conclusion:
FALTA