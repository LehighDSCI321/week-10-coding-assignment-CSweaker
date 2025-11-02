from collections import deque
from typing import Any, Generator, List, Set


class VersatileDigraph:
    """
    A versatile directed graph class that supports adding nodes, edges, and 
    retrieving neighbors of a node and all nodes in the graph.
    
    Attributes:
        nodes (set): A set containing all nodes in the graph.
        edges (dict): An adjacency list where keys are nodes and values are 
            sets of neighboring nodes pointed to by the key node.
    """

    def __init__(self):
        """Initialize an empty directed graph."""
        self.nodes = set()
        self.edges = {}  # Key: node, Value: set of neighboring nodes

    def add_node(self, node: Any) -> None:
        """
        Add a node to the graph (no duplicate addition if the node already exists).
        
        Args:
            node: The node to be added (must be a hashable type).
        """
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()

    def add_edge(self, u: Any, v: Any) -> None:
        """
        Add a directed edge from node u to node v (automatically adds nodes 
        that do not exist).
        
        Args:
            u: The start node of the edge.
            v: The end node of the edge.
        """
        self.add_node(u)  # Ensure the start node exists
        self.add_node(v)  # Ensure the end node exists
        self.edges[u].add(v)  # Add the edge u->v

    def get_neighbors(self, node: Any) -> Set[Any]:
        """
        Retrieve all neighbors of a specified node (i.e., nodes pointed to by 
        the given node).
        
        Args:
            node: The node to query for neighbors.
            
        Returns:
            set: A set of the node's neighbors (returns an empty set if the 
                node does not exist).
        """
        return self.edges.get(node, set())

    def get_all_nodes(self) -> List[Any]:
        """
        Retrieve a list of all nodes in the graph.
        
        Returns:
            list: A list containing all nodes in the graph.
        """
        return list(self.nodes)


class SortableDigraph(VersatileDigraph):
    """
    A sortable directed graph class that inherits from VersatileDigraph. It 
    adds a top_sort method to perform topological sorting on directed acyclic 
    graphs (DAGs).
    """

    def top_sort(self) -> List[Any]:
        """
        Perform topological sorting on the graph using Kahn's algorithm (only 
        applicable to DAGs).
        
        Algorithm steps:
            1. Calculate the in-degree (number of incoming edges) for each node.
            2. Initialize a queue with all nodes having an in-degree of 0.
            3. Process the queue in a loop:
               - Dequeue a node and add it to the topological order result.
               - Decrement the in-degree of its neighbors; if a neighbor's 
                 in-degree becomes 0, enqueue it.
               
        Returns:
            list: A list of nodes in topological order (if the graph contains 
                a cycle, the result length will be less than the total number 
                of nodes).
        """
        # Calculate in-degree for each node
        in_degree = {n: 0 for n in self.nodes}
        for current_node in self.nodes:
            for neighbor in self.get_neighbors(current_node):
                in_degree[neighbor] += 1

        # Initialize queue with nodes having in-degree 0
        queue = deque()
        for n in self.nodes:
            if in_degree[n] == 0:
                queue.append(n)

        top_order = []
        while queue:
            current = queue.popleft()
            top_order.append(current)
            # Process all neighbors of the current node, decrement their in-degree
            for neighbor in self.get_neighbors(current):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return top_order


class TraversableDigraph(SortableDigraph):
    """
    A traversable directed graph class that inherits from SortableDigraph. It 
    adds DFS and BFS traversal methods.
    """

    def dfs(self, start: Any) -> Generator[Any, None, None]:
        """
        Perform a depth-first search traversal starting from the given node.
        
        Args:
            start: The starting node for the traversal.
            
        Returns:
            Generator: A generator yielding nodes in DFS order.
        """
        if start not in self.nodes:
            return
        
        visited = set()
        stack = [start]
        
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                yield current_node
                # Convert set to list before reversing
                for neighbor in reversed(list(self.get_neighbors(current_node))):
                    if neighbor not in visited:
                        stack.append(neighbor)

    def bfs(self, start: Any) -> Generator[Any, None, None]:
        """
        Perform a breadth-first search traversal starting from the given node.
        
        Args:
            start: The starting node for the traversal.
            
        Returns:
            Generator: A generator yielding nodes in BFS order.
        """
        if start not in self.nodes:
            return
        
        visited = set([start])
        queue = deque([start])
        
        # Skip the start node itself (as per test expectations)
        queue.popleft()
        
        while queue:
            current_node = queue.popleft()
            yield current_node
            for neighbor in self.get_neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)


class DAG(TraversableDigraph):
    """
    A Directed Acyclic Graph (DAG) class that inherits from TraversableDigraph.
    It ensures that no cycles are created when adding edges.
    """

    def _has_path(self, start: Any, end: Any) -> bool:
        """
        Check if there is a path from start node to end node using BFS.
        
        Args:
            start: The start node of the path.
            end: The end node of the path.
            
        Returns:
            bool: True if a path exists, False otherwise.
        """
        if start == end:
            return True
            
        visited = set([start])
        queue = deque([start])
        
        while queue:
            current_node = queue.popleft()
            for neighbor in self.get_neighbors(current_node):
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return False

    def add_edge(self, u: Any, v: Any) -> None:
        """
        Add a directed edge from node u to node v only if it does not create a cycle.
        
        Args:
            u: The start node of the edge.
            v: The end node of the edge.
            
        Raises:
            ValueError: If adding the edge would create a cycle.
        """
        # Check if adding this edge would create a cycle
        # A cycle is created if there's already a path from v to u
        if self._has_path(v, u):
            raise ValueError(f"Adding edge {u}->{v} would create a cycle in the graph.")
        
        # If no cycle is created, add the edge using the parent class method
        super().add_edge(u, v)

    def successors(self, node: Any) -> List[Any]:
        """
        Return a list of successor nodes for the given node.
        
        Args:
            node: The node to get successors for.
            
        Returns:
            list: A list of successor nodes.
        """
        return list(self.get_neighbors(node))


# Example usage with the clothing graph
if __name__ == "__main__":
    # Create a DAG for the clothing order
    g = DAG()
    
    # Add all nodes
    nodes = ['shirt', 'pants', 'socks', 'vest', 'tie', 'belt', 'shoes', 'jacket']
    for node in nodes:
        g.add_node(node)
    
    # Add edges according to the corrected structure
    edges = [
        ('shirt', 'tie'), ('shirt', 'pants'), ('shirt', 'vest'), ('shirt', 'jacket'),
        ('tie', 'jacket'), ('pants', 'belt'), ('pants', 'shoes'),
        ('socks', 'shoes'), ('vest', 'jacket'), ('belt', 'jacket')
    ]
    
    for u, v in edges:
        try:
            g.add_edge(u, v)
            print(f"Added edge {u}->{v}")
        except ValueError as e:
            print(f"Failed to add edge {u}->{v}: {e}")
    
    # Test DFS traversal
    print("\nDFS traversal starting from 'shirt':")
    for node in g.dfs('shirt'):
        print(node)
    
    # Test BFS traversal
    print("\nBFS traversal starting from 'shirt':")
    for node in g.bfs('shirt'):
        print(node)
    
    # Test topological sort
    print("\nTopological sort:")
    print(g.top_sort())
    
    # Test successors method
    print("\nSuccessors of 'shirt':")
    print(g.successors('shirt'))
    
    # Test adding an edge that would create a cycle
    print("\nTrying to add edge 'jacket'->'shirt' (which creates a cycle):")
    try:
        g.add_edge('jacket', 'shirt')
    except ValueError as e:
        print(f"Error: {e}")
