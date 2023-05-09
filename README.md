# Investigating the Impact of Different Agent Exploration Strategies on Performance in the Ms. Pac-Man Game Environment

This project explores the performance of three different agent exploration strategies in the Ms. Pac-Man game environment: genetic algorithms, reactive agents, and A* search. The pacman enviroment simulator is from https://github.com/jspacco/pac3man - this is a Python 3 version of a simulator that was developed to teach AI at the University of California, Berkeley a few years ago. There are loads of details about it at http://ai.berkeley.edu/project_overview.html  

## Project Overview

The goal of this project is to understand how different agent exploration strategies affect performance in the Ms. Pac-Man game. I implemented and compared the following agents:

1. Genetic Algorithm Agent: This agent uses a genetic algorithm to evolve a population of candidate solutions (game-playing strategies) over multiple generations.
2. Reactive Agent: This agent takes actions based on the current state of the game without any learning or adaptation. The agent moves away from ghosts, moves toward food pellets, and takes random legal actions when no food pellet is nearby.
3. A* Agent: This agent uses the A* search algorithm to find the shortest path to a target, such as a food pellet.

I evaluated the performance of each agent using average and best scores across multiple runs.

## Repository Structure

- `search/`: Contains all the agents and the file to run the test.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## How to Run

To run the project, navigate to the `search/` directory and execute the runGames.py script, for different agent you wish to test, you may need to change the agent name at the start of the file and uncomment/comment some part of the code. Remember: NewAgent1 class in newAgents.py represents genetic algorithm agent.


## Author

Yiming Yuan

Coursework for COMP3004 Designing Intelligent Agents
