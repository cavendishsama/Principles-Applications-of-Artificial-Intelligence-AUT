import numpy as np
from state import next_state, solved_state
from location import next_location
from collections import OrderedDict
from location import solved_location
import signal
import heapq
from time import time
def signal_handler(signum, frame):
    raise Exception("Timed out!")


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube_state using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube_state.
        init_location (numpy.array): Initial location of the little cube_states.
        method (str): Name of the search algorithm.
 
    Returns:
        list: The sequence of actions needed to solve the Rubik's cube_state.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cube_states when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    # Defining Heuristic function
    def heuristic(current_location):
        goal_state = solved_location()

        distance = 0
        current_location = np.array(current_location)

        # It iterates through the eight cubes of the 2x2 Rubik's Cube (range(8)).
        for i in range(8):
            for level in range(2):
                for col in range(2):
                    for row in range(2):

                        if (current_location[row, col, level] == goal_state[row, col, level]):
                            return distance / 4

                        else:
                            for _level in range(2):
                                for _cow in range(2):
                                    for _row in range(2):
                                        if (current_location[_row, _cow, _level] == goal_state[row, col, level]):
                                            # Calculating the Manhattan distance 
                                            distance += np.abs(_row - row) + np.abs(_cow - col) + np.abs(+ _level - level)

            return distance / 4



    if method == 'Random':
        return list(np.random.randint(1, 12+1, 10))
    
    elif method == 'IDS-DFS':
        
        class Node:
            cube_state = None
            parent = None
            move = None

        limit = 1
        moves = []  # Moves done to reach the final state
        next_node = OrderedDict()
        Final_state = solved_state()

        if((init_state == Final_state).all()):
            print("Already final_state :/")
            return

        while True:
            root_node = Node()
            root_node.cube_state = init_state
            next_node[hash(root_node.cube_state.tobytes())] = root_node
        
            while bool(next_node):
                current_node = Node()
                _, current_node = next_node.popitem(last=False)
                
                # finding moves after solving the 
                if ((current_node.cube_state == Final_state).all()):

                    node_search = current_node
                    while(node_search.move != None):
                        moves.append(node_search.move)
                        node_search = node_search.parent
                    
                    return list(reversed(moves))
                    

                for i in range(12):
                    chilld_node = Node()
                    chilld_node.cube_state = next_state(current_node.cube_state, i + 1)
                
                    chilld_node.parent = current_node
                    chilld_node.move = i + 1
                    next_node[hash(chilld_node.cube_state.tobytes())] = chilld_node

                limit = limit + 1
    
    elif method == 'A*':
        
        start_time = time()

        class Node_A_star:
            cube = None
            cost = 0
            parent = None
            move = None
            heuristic = None
            location = None

            def __lt__(self, other):
                return (self.cost + self.heuristic) < (other.cost + other.heuristic)


        final_state = solved_state()
        cost_limit = 1
        Explored_nodes = 0
        Expanded_nodes = 0
        depth = 0
        moves = []
        frontier = []  # Use a list for the priority queue
        # heapq.heapify(frontier)  # Priority queue using heapq

        initial = Node_A_star()
        initial.cube = init_state
        initial.cost = 0
        initial.location = init_location
        initial.heuristic = heuristic(init_location)

        heapq.heappush(frontier, (initial.cost + initial.heuristic, initial))
        visited = set()

        while frontier:
            _, current_node = heapq.heappop(frontier)

            # Check if the Node is visited or not
            if hash(current_node.cube.tobytes()) in visited:
                continue

            # Current node is designated as visited
            visited.add(hash(current_node.cube.tobytes()))

            Explored_nodes += 1

            if hash(current_node.cube.tobytes()) == hash(final_state.tobytes()):
                # Goal state reached
                node_search = current_node
                while node_search.move is not None:
                    moves.append(node_search.move)
                    node_search = node_search.parent
                
                end_time = time()
                print("Expanded Nodes:", Expanded_nodes)
                print("Explored Nodes:", Explored_nodes)
                print("Depth of Final graph:", depth)
                print("Elapsed Time:", end_time - start_time)
                return list(reversed(moves))

            if current_node.cost + 1 <= cost_limit:
                child_cost = current_node.cost + 1
                for i in range(12):
                    Expanded_nodes += 1
                    new_node = Node_A_star()
                    new_node.cube = next_state(current_node.cube, i + 1)
                    new_node.cost = child_cost
                    new_node.location = next_location(current_node.location, i + 1)
                    new_node.heuristic = heuristic(new_node.location)
                    new_node.parent = current_node
                    new_node.move = i + 1
                    heapq.heappush(frontier, (new_node.cost + new_node.heuristic, new_node))

            cost_limit += 1

    elif method == 'BiBFS':
        ...
    
    else:
        return []