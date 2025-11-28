# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue
import math


"""
Instead of creating your own new class of objects, you can just use tuples. Tuples are always hashable, and the rules for sorting of tuples are listed here. Note that the syntax for creating single-element tuples is weird (see this page), and the easiest way to concatenate two tuples is to use the + operator (see this page); we recommend you test your ideas in an interactive window before writing your submitted.py.
"""

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    
    # define nodes
    start_node = maze.start
    curr_node = start_node
    end_node = maze.waypoints[0]
    
    priority_queue = queue.Queue()
    priority_queue.put(start_node)
    visited_nodes = set()
    solution_path = []
    backtrace = {}
    
    # loop through until queue is done
    while (not priority_queue.empty()) : 
        curr_node = priority_queue.get()
        
        if curr_node == end_node:  
            while curr_node != start_node: #backtrace here
                # update curr_node with its predecessor
                solution_path = [curr_node] + solution_path
                curr_node = backtrace[curr_node]
                
            solution_path = [start_node] + solution_path
            return solution_path
          
        # loop through the neighbors
        visited_nodes.add(curr_node)
        for neighbor in maze.neighbors_all(curr_node[0], curr_node[1]):            
            # if neigbor isn't in the queue to check add it to the queue for the backtrace
            if neighbor not in visited_nodes:
                priority_queue.put(neighbor)
                visited_nodes.add(neighbor)
                backtrace[neighbor] = curr_node
    
    # in case nothing found
    return []

"""
    Returns the chebyshev distance

    @param curr_node: the current node we are on
    @param destination: the node we wnat to reach

    @return chebyshev_distance: the chebyshev_distance between the current node and the destination node
"""
def chebyshev_distance(curr_node, destination):
        return max(abs(curr_node[0] - destination[0]), abs(curr_node[1] - destination[1]))
        
"""
g = the movement cost to move from the starting point to a given square on the grid, following the path generated to get there.

h = the estimated movement cost to move from that given square on the grid to the final destination. This is often referred to as the heuristic, which is nothing but a kind of smart guess. We really don’t know the actual distance until we find the path, because all sorts of things can be in the way (walls, water, etc.). If the heuristic is admissible, then A* search is optimal.

"""
def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    #define nodes
    start_node = maze.start
    curr_node = start_node
    end_node = maze.waypoints[0]
    
    priority_queue = queue.PriorityQueue()
    priority_queue.put((0, start_node[0], start_node[1]))
    visited_nodes = set()
    solution_path = []
    backtrace = {}
    
    # fill maze_nodes
    maze_nodes = []
    for r in range(maze.size.y):
        row = []
        for c in range(maze.size.x):
            cell = [float("inf"), float("inf"), 0]
            row.append(cell)
        maze_nodes.append(row)
    # set starting nodes to 0 path
    maze_nodes[start_node[0]][start_node[1]][0] = 0
    maze_nodes[start_node[0]][start_node[1]][1] = 0
    maze_nodes[start_node[0]][start_node[1]][2] = 0
    
    # backtrace
    while not priority_queue.empty():
        curr_node, h, path = priority_queue.get()
        visited_nodes.add((h, path))
        neighbors = maze.neighbors_all(h, path)
        
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                
                if end_node[1] == neighbor[1] and end_node[0] == neighbor[0]:
                    
                    backtrace[neighbor] = (h, path)
                    
                    while not (neighbor[1] == start_node[1] and neighbor[0] == start_node[0]):
                        solution_path = [neighbor] + solution_path
                        neighbor = backtrace[neighbor]
                        
                    solution_path = [(start_node[0], start_node[1])] + solution_path
                    return solution_path
                
                neighbor_total_cost = (maze_nodes[h][path][1] + 1) + (chebyshev_distance(end_node, neighbor))
                                
                if neighbor_total_cost < maze_nodes[neighbor[0]][neighbor[1]][0]: #if valid to move to
                    priority_queue.put((neighbor_total_cost, neighbor[0], neighbor[1]))
                    
                    maze_nodes[neighbor[0]][neighbor[1]][1] = maze_nodes[h][path][1] + 1
                    maze_nodes[neighbor[0]][neighbor[1]][2] = chebyshev_distance(end_node, neighbor)
                    maze_nodes[neighbor[0]][neighbor[1]][0] = neighbor_total_cost

                    backtrace[neighbor] = (h, path)
    
    return [] # if nothing found


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    # starting vals
    start_node = maze.start
    curr_node = start_node
    end_node = maze.waypoints[0]

    nodes = list(maze.waypoints)
    start = maze.start
    solution_path = []
    neighbors = []
    visited_nodes = set()
    

    max_distance = 0
    for node in nodes:
        h = chebyshev_distance(start, node)
        
        if h > max_distance: # not admissable
            max_distance = h
            end_node = node
            
    nodes.remove(end_node)
    neighbors.append(end_node)
    current_node = end_node
        
    # get all neighbors
    while not len(nodes) == 0:
        temp_h = math.inf
        neighbor_node = None
        
        for node in nodes:
            h = chebyshev_distance(current_node, node)
            
            if h < temp_h and h != 0:
                temp_h = h
                neighbor_node = node

        neighbors.append(neighbor_node)
        current_node = neighbor_node
        nodes.remove(neighbor_node)
        
    neighbors.append(maze.start)
    
    # check all neighbors
    while len(neighbors) > 1:
        start_node = neighbors.pop()
        finish_node_curr = neighbors[-1]
        
        visited_nodes = set()
        priority_queue = queue.PriorityQueue()
        priority_queue.put((0, [start_node]))
        
        while not priority_queue.empty():
            path = priority_queue.get()[1]
            
            curr_node = path[-1] 
            if (curr_node[0], curr_node[1]) == finish_node_curr:
                break
                
            curr_neighbors = maze.neighbors_all(curr_node[0], curr_node[1])
            for neighbor in curr_neighbors:
                if neighbor not in visited_nodes:
                    h = chebyshev_distance(neighbor, finish_node_curr) 
                    priority_queue.put((h + len(path), path + [neighbor])) # add total cost (f)
              
                    visited_nodes.add(neighbor)
        
        
        solution_path = solution_path + path
        solution_path.pop()
        
    solution_path = solution_path + [end_node]
    return solution_path
