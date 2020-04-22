"""
graphs.py

Directed graph algorithm implementations.
"""


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle, assuming that no cycle already exists in
    the graph represented by 'connections'.
    
    :param connections: List of connections, with each connection a tuple of form (sender, receiver)
    :param test: Newly added connection, represented by a tuple of form (sender, receiver)
    """
    i, o = test
    if i == o: return True
    
    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i: return True
                visited.add(b)
                num_added += 1
        if num_added == 0: return False


def required_for_output(inputs: set, outputs: set, connections: dict):
    """
    Determine which nodes and connections are needed to compute the final output. It is considered that only paths
     starting at the inputs and ending at the outputs are relevant. This decision is made since a node bias can
     substitute for a 'floating' node (i.e. node with no input and constant output).
     
    This algorithm works in two steps:
     * A growing-phase; adding nodes that are connected (albeit indirectly) to the outputs of the networks
     * A pruning-phase; removing dead hidden nodes that do not contain both an ingoing and an outgoing connection
    
    :note: It is assumed that the input identifier set and the node identifier set are disjoint. By convention, the
           output node ids are always the same as the output index.
    
    :param inputs: Set of the used input identifiers
    :param outputs: Set of all the output node identifiers
    :param connections: Dictionary of genome connections
    :return: Sets of: used inputs, used hidden nodes, used output nodes, remaining connections
    """
    # Only consider the enabled connections
    enabled_conn = {k: c for k, c in connections.items() if c.enabled}
    non_recur_enabled_conn = {(a, b): c for (a, b), c in enabled_conn.items() if a != b}
    
    # Check if both in and outputs are used
    used_inputs = {a for (a, _) in enabled_conn.keys() if a < 0}
    if not used_inputs: return set(), set(), outputs, dict()
    used_outputs = {b for (_, b) in enabled_conn.keys() if b in outputs}
    if not used_outputs: return set(), set(), outputs, dict()
    
    # Growing-phase
    added_nodes = {i for i in outputs}
    used_nodes = {i for i in outputs}
    while added_nodes:
        # Find all nodes sending to one of the used nodes
        nodes = {a for (a, b) in non_recur_enabled_conn.keys() if b in used_nodes}
        
        # Update the two sets
        added_nodes = nodes - used_nodes
        used_nodes.update(nodes)
    
    # Get all the connections that are used
    used_conn = {(a, b): c for (a, b), c in enabled_conn.items() if (a in used_nodes) and (b in used_nodes)}
    non_recur_used_conn = {(a, b): c for (a, b), c in used_conn.items() if a != b}
    
    # Pruning-phase
    removed_nodes = {True}  # dummy
    while removed_nodes:
        # Find all nodes that do both send and receive, or are input or output nodes
        sending = {a for (a, _) in non_recur_used_conn.keys() if a in used_nodes}
        receiving = {b for (_, b) in non_recur_used_conn.keys() if b in used_nodes}
        nodes = {n for n in used_nodes if (n in sending & receiving) or (n in inputs | outputs)}
        
        # Check if any nodes are removed and update the current sets
        removed_nodes = used_nodes - nodes
        used_nodes = nodes
        
        # Update the used connections
        used_conn = {(a, b): c for (a, b), c in enabled_conn.items() if (a in used_nodes) and (b in used_nodes)}
        non_recur_used_conn = {(a, b): c for (a, b), c in used_conn.items() if a != b}
    
    # Test if there are used connection remaining
    if not used_conn:
        return set(), set(), outputs, dict()
    
    # Connected network, return all the used nodes and connections
    used_inp = {n for n in used_nodes if n < 0}
    used_out = outputs  # All outputs are always considered used
    used_hid = {n for n in used_nodes if n not in inputs | outputs}
    
    # Check (again) if both in and outputs are used
    used_inputs = {a for (a, _) in enabled_conn.keys() if a < 0}
    if not used_inputs: return set(), set(), outputs, dict()
    used_outputs = {b for (_, b) in enabled_conn.keys() if b in outputs}
    if not used_outputs: return set(), set(), outputs, dict()
    
    # Valid genome, return
    return used_inp, used_hid, used_out, used_conn
