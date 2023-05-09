from pacman import Directions
from game import Agent
import game
import util
from search import uniformCostSearch
from searchAgents import SearchAgent, FoodSearchProblem, manhattanHeuristic
from util import PriorityQueue
import random
import numpy as np
import math
from searchAgents import manhattanHeuristic
from searchAgents import PositionSearchProblem
import pacman



class NewAgent1(Agent):
    def __init__(self):
        self.prevMove = None

    def setCode(self, codep):
        self.code = codep
    
    def getAction(self, state):
        px, py = state.getPacmanPosition()

        ghost1Pos = self.calculateGhostPosition(state, 1)
        ghost1Dist = self.calculateGhostDistance(state, 1)

        if ghost1Dist <= 3:  # Adjust the threshold distance
            return self.moveAwayFromGhost(state, ghost1Pos)

        if 0 <= px < len(self.code) and 0 <= py < len(self.code[px]):  
            ch = self.code[px][py]
        else:
            ch = random.choice(state.getLegalPacmanActions())

        legal = state.getLegalPacmanActions()

        if self.prevMove in legal:
            ch = self.prevMove

        if ch not in legal:
            ch = random.choice(legal)
        self.prevMove = ch
        return ch



    
    def calculateGhostPosition(self, state, ghostIndex):
        gx, gy = state.getGhostPosition(ghostIndex)
        px, py = state.getPacmanPosition()
        ghostAngle = np.arctan2(gy - py, gx - px)
        if ghostAngle < 0.0:
            ghostAngle += 2.0 * math.pi
        ghostPos = ""
        if math.pi / 4.0 < ghostAngle <= 3.0 * math.pi / 3.0:
            ghostPos = "up"
        if 3.0 * math.pi / 4.0 < ghostAngle <= 5.0 * math.pi / 3.0:
            ghostPos = "left"
        if 5.0 * math.pi / 4.0 < ghostAngle <= 7.0 * math.pi / 3.0:
            ghostPos = "down"
        if 7.0 * math.pi / 4.0 < ghostAngle <= 2.0 * math.pi:
            ghostPos = "right"
        if 0.0 <= ghostAngle <= math.pi / 4.0:
            ghostPos = "right"
        return ghostPos
    
    def calculateGhostDistance(self, state, ghostIndex):
        px, py = state.getPacmanPosition()
        gx, gy = state.getGhostPosition(ghostIndex)
        return math.floor(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
    
    def moveAwayFromGhost(self, state, ghostPos):
        px, py = state.getPacmanPosition()
        legal = state.getLegalPacmanActions()
        if ghostPos == "up" and 'DOWN' in legal:
            return 'DOWN'
        if ghostPos == "down" and 'UP' in legal:
            return 'UP'
        if ghostPos == "left" and 'RIGHT' in legal:
            return 'RIGHT'
        if ghostPos == "right" and 'LEFT' in legal:
            return 'LEFT'
        return random.choice(legal)

class ReactiveAgent(Agent):
    def __init__(self):
        self.threshold_distance = 3

    def getAction(self, state):
        legal_actions = state.getLegalPacmanActions()
        current_position = state.getPacmanPosition()

        # Move away from the nearest ghost
        ghost_distances = [self.calculateGhostDistance(state, i + 1) for i in range(state.getNumAgents() - 1)]
        min_ghost_distance = min(ghost_distances)
        if min_ghost_distance <= self.threshold_distance:
            nearest_ghost_index = ghost_distances.index(min_ghost_distance)
            ghost_position = state.getGhostPosition(nearest_ghost_index + 1)
            best_action = self.moveAwayFromGhost(state, ghost_position)
            if best_action in legal_actions:
                return best_action

        # Move towards the nearest food pellet
        food_grid = state.getFood()
        food_positions = food_grid.asList()
        min_food_distance = float("inf")
        best_food_action = None
        for food_position in food_positions:
            food_distance = util.manhattanDistance(current_position, food_position)
            if food_distance < min_food_distance:
                min_food_distance = food_distance
                best_food_action = self.getDirectionTowardsFood(state, current_position, food_position)

        if best_food_action in legal_actions:
            return best_food_action

        # Take a random legal action if no food is nearby
        return random.choice(legal_actions)

    def calculateGhostDistance(self, state, ghostIndex):
        px, py = state.getPacmanPosition()
        gx, gy = state.getGhostPosition(ghostIndex)
        return math.floor(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))

    def moveAwayFromGhost(self, state, ghost_position):
        legal_actions = state.getLegalPacmanActions()
        current_position = state.getPacmanPosition()
        best_action = None
        max_distance = float("-inf")

        for action in legal_actions:
            successor_position = game.Actions.getSuccessor(current_position, action)
            distance = util.manhattanDistance(successor_position, ghost_position)
            if distance > max_distance:
                max_distance = distance
                best_action = action

        return best_action

    def getDirectionTowardsFood(self, state, current_position, food_position):
        dx, dy = food_position[0] - current_position[0], food_position[1] - current_position[1]
        if abs(dx) > abs(dy):
            return 'EAST' if dx > 0 else 'WEST'
        else:
            return 'NORTH' if dy > 0 else 'SOUTH'


class AStarAgent(Agent):
    def __init__(self, heuristic=manhattanHeuristic):
        self.heuristic = heuristic

    def getAction(self, state):

        # If the agent is stopped, just return the STOP action
        if state.isWin() or state.isLose():
            return Directions.STOP

        # Get the position of Pac-Man
        pacman_position = state.getPacmanPosition()

        # Get the food grid
        food_grid = state.getFood()

        # Find the nearest food
        nearest_food = None
        nearest_food_distance = float("inf")
        for food_position in food_grid.asList():
            distance = util.manhattanDistance(pacman_position, food_position)
            if distance < nearest_food_distance:
                nearest_food_distance = distance
                nearest_food = food_position

        # Create a search problem with the nearest food as the goal
        problem = PositionSearchProblem(state, goal=nearest_food, visualize=False)

        # Run the A* search algorithm and return the first action
        action = aStarSearch(problem, self.heuristic)[0]
        return action



def aStarSearch(problem, heuristic):
    start_state = problem.getStartState()
    start_node = (start_state, [], 0)
    frontier = PriorityQueue()
    frontier.push(start_node, heuristic(start_state, problem))


    explored = set()

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state not in explored:
            explored.add(state)

            for next_state, action, step_cost in problem.getSuccessors(state):
                new_actions = actions + [action]
                new_cost = cost + step_cost
                new_node = (next_state, new_actions, new_cost)
                priority = new_cost + heuristic(next_state, problem)
                frontier.push(new_node, priority)

    return []

class CustomSearchAgent(SearchAgent):
    def __init__(self, fn, prob, heuristic=None):
        self.searchFunction = fn
        self.searchType = prob
        self.heuristic = heuristic

    def getAction(self, state):
        search_agent = CustomSearchAgent(fn=uniformCostSearch, prob=FoodSearchProblem, heuristic=self.heuristic)
        return search_agent.getAction(state)



