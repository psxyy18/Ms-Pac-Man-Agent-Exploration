import matplotlib.pyplot as plt
from pacman import *
import ghostAgents
import layout as layout_module  
import textDisplay
import graphicsDisplay
import copy
import numpy as np
from pprint import pprint
import sys
from newAgents import NewAgent1
from newAgents import ReactiveAgent
from newAgents import AStarAgent
from searchAgents import manhattanHeuristic
import random
from game import Agent
from search import aStarSearch
from searchAgents import PositionSearchProblem




## set up the parameters to newGame
numtraining = 0
timeout = 30
beQuiet = True
layout=layout.getLayout("mediumClassic")
#pacmanType = loadAgent("NewAgent1")
pacmanType = ReactiveAgent
numGhosts = 1
ghosts = [ghostAgents.RandomGhost(i+1) for i in range(numGhosts)]
catchExceptions=True

def run(code, noOfRuns, game_layout):
    rules = ClassicGameRules(timeout)
    games = []
    if beQuiet:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        timeInterval = 0.001
        textDisplay.SLEEP_TIME = timeInterval
        gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    for gg in range(noOfRuns):
        thePacman = pacmanType()
        if isinstance(thePacman, NewAgent1):  
            thePacman.setCode(code)
        game = rules.newGame(game_layout, thePacman, ghosts, gameDisplay,
                     beQuiet, catchExceptions)


        game.run()
        games.append(game)
    scores = [game.state.getScore() for game in games]
    return sum(scores) / float(len(scores))

####### genetic algorithm

options = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
    
def mutate(parentp,numberOfMutations=10):
    parent = copy.deepcopy(parentp)
    for _ in range(numberOfMutations):
        xx = random.randrange(19)
        yy = random.randrange(10)
        parent[xx][yy] = random.choice(options)
    return parent

def runGA(popSiz=20, timescale=10, numberOfRuns=3, tournamentSize=4, numberOfMutations=3, blockSize=2, layout_name="originalClassic"):

    ## create random initial population
    game_layout = layout_module.getLayout(layout_name)  

    population = []
    for _ in range(popSiz):
        program = np.empty((19, 10), dtype=object)
        for xx in range(19):
            for yy in range(10):
                program[xx][yy] = random.choice(options)
        population.append(program)
            
    print("Beginning Evolution")
    averages = []
    bests = []
    for _ in range(timescale):
        ## evaluate population
        fitness = []
        for pp in population:
            print(".",end="",flush=True)
            fitness.append(run(pp, numberOfRuns, game_layout)) 
        print("\n******")
        print(fitness)
        averages.append(1000+sum(fitness)/popSiz)
        print("av ",1000+sum(fitness)/popSiz)
        bests.append(1000+max(fitness))
        print("max ",1000+max(fitness))

        popFitPairs = list(zip(population,fitness))
        newPopulation = []
        for _ in range(popSiz-1):
                # select a parent from a "tournament"
                tournament = random.sample(popFitPairs,tournamentSize)
                parent = max(tournament,key=lambda x:x[1])[0]
                # mutate the parent
                child = mutate(parent)
                newPopulation.append(child)
                ## CROSSOVER PARENTS
                parent2 = max(random.sample(popFitPairs, tournamentSize), key=lambda x: x[1])[0]
                child = crossover(parent, parent2)
                newPopulation.append(child)

        ## KEEP BEST POPULATION MEMBER
        best_individual = max(popFitPairs, key=lambda x: x[1])[0]
        newPopulation.append(best_individual)

        population = copy.deepcopy(newPopulation)
    print(averages)
    print(bests)
    plot_scores(averages, bests)
    return averages[-1], bests[-1]

def random_search(num_searches):
    best_score = -1
    best_params = None

    for i in range(num_searches):
        print(f"Search {i+1}/{num_searches}")

        # Randomly sample parameter values
        popSiz = random.choice(range(10, 31, 10))
        timescale = random.choice(range(10, 51, 10))
        numberOfRuns = random.choice(range(1, 4))
        tournamentSize = random.choice(range(2, 6))
        numberOfMutations = random.choice(range(1, 11))
        blockSize = random.choice(range(1, 6))
        layout_name = random.choice(["mediumClassic", "smallClassic", "testClassic"])



        # Run the genetic algorithm with the sampled parameter values
        layout = layout_module.getLayout(layout_name) 

        avg_score, _ = runGA(popSiz, timescale, numberOfRuns, tournamentSize, numberOfMutations, blockSize, layout_name)


        # Update the best parameters if the current parameters give a higher score
        if avg_score > best_score:
            best_score = avg_score
            best_params = {
                "popSiz": popSiz,
                "timescale": timescale,
                "numberOfRuns": numberOfRuns,
                "tournamentSize": tournamentSize,
                "numberOfMutations": numberOfMutations,
                "blockSize": blockSize,
                "layout_name": layout_name
            }

    return best_params, best_score



    ##  PLOT averages AND bests
def plot_scores(averages, bests):
    plt.plot(averages, label="Average")
    plt.plot(bests, label="Best")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Pac-Man Genetic Algorithm Performance")
    plt.legend()
    plt.show()

def crossover(parent1, parent2, blockSize=5):
    child = np.empty_like(parent1)
    for i in range(0, child.shape[0], blockSize):
        for j in range(0, child.shape[1], blockSize):
            selected_parent = parent1 if random.random() < 0.5 else parent2
            child[i:i+blockSize, j:j+blockSize] = selected_parent[i:i+blockSize, j:j+blockSize]
    return child



def runTest():
    program = np.empty((19,10),dtype=object)
    for xx in range(19):
        for yy in range(10):
            program[xx][yy] = Directions.EAST
    
    run(program,1)

def run_reactive_agent_and_plot(agent, num_runs, layout_name):
    game_layout = layout_module.getLayout(layout_name)
    scores = []

    for i in range(num_runs):
        score = run(None, 1, game_layout)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    best_score = max(scores)

    plt.plot(scores, label="Reactive Agent")
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.title("Pac-Man Reactive Agent Performance")
    plt.legend()
    plt.show()

    return avg_score, best_score

import matplotlib.pyplot as plt

def run_astar_agent_and_plot(agent, num_runs, layout_name):
    game_layout = layout_module.getLayout(layout_name)
    scores = []

    for i in range(num_runs):
        score = run(None, 1, game_layout)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    best_score = max(scores)

    plt.plot(scores, label="A* Agent")
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.title("Pac-Man A* Agent Performance")
    plt.legend()
    plt.show()

    return avg_score, best_score


#runTest()    

#runGA(popSiz=20, timescale=10, numberOfRuns=3, tournamentSize=4, numberOfMutations=3, blockSize=2, layout_name="originalClassic")

def main(agent_type):
    num_runs = 5  # Number of times to run the chosen agent

    if agent_type == 'reactive':
        layout_name = 'mediumClassic'
        avg_score, best_score = run_reactive_agent_and_plot(ReactiveAgent, num_runs, layout_name)
        print(f"Reactive Agent - Average score: {avg_score}, Best score: {best_score}")

    elif agent_type == 'genetic':
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}")

            # my own best parameters
            best_params = {
                'popSiz': 30,
                'timescale': 30,
                'numberOfRuns': 3,
                'tournamentSize': 3,
                'numberOfMutations': 9,
                'blockSize': 3,
                'layout_name': 'mediumClassic'
            }

            # Run the genetic algorithm using the best parameters
            avg_score, best_score = runGA(
                popSiz=best_params['popSiz'],
                timescale=best_params['timescale'],
                numberOfRuns=best_params['numberOfRuns'],
                tournamentSize=best_params['tournamentSize'],
                numberOfMutations=best_params['numberOfMutations'],
                blockSize=best_params['blockSize'],
                layout_name=best_params['layout_name']
            )

            print(f"Run {i+1} - Average score with the best parameters: {avg_score}")
            print(f"Run {i+1} - Best score with the best parameters: {best_score}")
            print()

    elif agent_type == 'astar':
        layout_name = 'mediumClassic'
        avg_score, best_score = run_astar_agent_and_plot(AStarAgent, num_runs, layout_name)
        print(f"A* Agent - Average score: {avg_score}, Best score: {best_score}")


    else:
        print("Invalid agent type. Choose 'reactive', 'genetic', or 'astar'.")

if __name__ == "__main__":
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'reactive'
    #main(agent_type)
    main('reactive')

"""#code for reactive agent and genetic algorithm
if __name__ == "__main__":
    num_runs = 5  # Number of times to run the reactive agent
    layout_name = "mediumClassic"

    if use_reactive_agent:
        avg_score, best_score = run_reactive_agent_and_plot(num_runs, layout_name)
        print(f"Reactive Agent - Average score: {avg_score}, Best score: {best_score}")
    else:
        # (below are code for Genetic Algorithm)
         for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}")

            # my own best parameters
            best_params = {
                'popSiz': 30,
                'timescale': 30,
                'numberOfRuns': 3,
                'tournamentSize': 3,
                'numberOfMutations': 9,
                'blockSize': 3,
                'layout_name': 'mediumClassic'
            }

            # Run the genetic algorithm using the best parameters
            avg_score, best_score = runGA(
                popSiz=best_params['popSiz'],
                timescale=best_params['timescale'],
                numberOfRuns=best_params['numberOfRuns'],
                tournamentSize=best_params['tournamentSize'],
                numberOfMutations=best_params['numberOfMutations'],
                blockSize=best_params['blockSize'],
                layout_name=best_params['layout_name']
            )

            results.append((avg_score, best_score))

            print(f"Run {i+1} - Average score with the best parameters: {avg_score}")
            print(f"Run {i+1} - Best score with the best parameters: {best_score}")
            print()

        # Print the results of all runs
        print("Results of all runs:")
        for i, (avg_score, best_score) in enumerate(results, 1):
            print(f"Run {i} - Average score: {avg_score}, Best score: {best_score}")

        avg_scores = [avg_score for avg_score, _ in results]
        best_scores = [best_score for _, best_score in results]

        avg_of_avg_scores = sum(avg_scores) / len(avg_scores)
        avg_of_best_scores = sum(best_scores) / len(best_scores)

        print(f"Average of average scores: {avg_of_avg_scores}")
        print(f"Average of best scores: {avg_of_best_scores}")
        """
