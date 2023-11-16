import numpy as np
from state import next_state, solved_state
from location import next_location
from collections import OrderedDict
import signal

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
            print("Already solved :/")
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
        ...

    elif method == 'BiBFS':
        ...
    
    else:
        return []