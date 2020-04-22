"""
genome_distance.py

Cache to calculate and store the genome-distances.
"""

from configs.genome_config import GenomeConfig
from population.utils.cache.node_comparing import NodeComparingCache


class GenomeDistanceCache(object):
    """Makes sure that redundant distance-computations will not occur. (e.g. d(1,2)==d(2,1))."""
    
    def __init__(self, config: GenomeConfig):
        self.distances = dict()
        self.kwargs = dict()  # Extra information about the distance-measure for the node
        self.config = config
        self.node_cache = NodeComparingCache()
    
    def __call__(self, genome0, genome1):
        """
        Calculate the genetic distances between two genomes and store in cache if not yet present.
        
        Distance is calculated as follows:
            1) Determine the distance between the nodes. Each node is compared with 'other' in sorted order. Two nodes
                are disjoint if they do not share the same type or have at least one disjoint input connection. If two
                nodes are not disjoint, a node-specific distance is measured. During this process, it is memorized which
                nodes share the same type and input-connections, hence, which keys are considered equal.
            2) Determine the distance between the connections. Two connections are disjoint if they do not share the
                same key. Note, however, the "sharing the same key" can vary due to the nodes that are considered equal.
        
        :note: Two nodes are NOT disjoint if:
            * They share the same key
            * They have the exact same type and inputs
        """
        # Used parameters
        distance = 0
        
        # Lowest key is always first, removes redundant distance checks
        g0, g1 = (genome0.key, genome1.key) if (genome0.key <= genome1.key) else (genome1.key, genome0.key)
        
        # Return distance if already in cache
        if (g0, g1) in self.distances.keys():
            return self.distances[g0, g1]
        
        # Distance to one-self is always equal to zero
        if g0 == g1:
            self.distances[g0, g1] = 0
            self.kwargs[g0, g1] = (0, 0)  # (disjoint_nodes, disjoint_connections)
            return self.distances[g0, g1]
        
        # Fetch the connections and nodes used by the genomes
        used_nodes0 = genome0.get_used_nodes()
        used_nodes1 = genome1.get_used_nodes()
        used_conn0 = genome0.get_used_connections()
        used_conn1 = genome1.get_used_connections()
        
        # Distance between the genomes is not yet calculated, start with calculating node-distance
        #  The node-distance is defined by the distance between comparable nodes and the number of disjoint nodes
        node_distance = dict()  # Dictionary mapping nodes (genome0, genome1) to their distance
        
        # Get all the distances of the comparable nodes (does not contain duplicates)
        for node0_id, node0 in used_nodes0.items():
            for node1_id, node1 in used_nodes1.items():
                comp, dist = self.node_cache(
                        node0=node0,
                        node1=node1,
                        conn0=used_conn0,
                        conn1=used_conn1,
                        cfg=self.config,
                )
                if comp: node_distance[node0_id, node1_id] = dist
        
        # Get all the disjoint nodes, these are the nodes not present in node_distance (were not comparable)
        disjoint_nodes = 0
        for node0_id in used_nodes0.keys():
            if not {node0_id for (a, _) in node_distance.keys() if a == node0_id}:
                disjoint_nodes += 1
        for node1_id in used_nodes1.keys():
            if not {node1_id for (_, b) in node_distance.keys() if b == node1_id}:
                disjoint_nodes += 1
        
        # The final node-distance is equal to linear sum of the average distance of the comparable nodes with the number
        #  of disjoint/excess nodes that are divided by the total number of nodes from the larger gene
        max_nodes = max(len(used_nodes0), len(used_nodes1), 1)
        if node_distance:
            distance += sum(node_distance.values()) / len(node_distance)
        distance += (self.config.compatibility_disjoint_node * disjoint_nodes) / max_nodes
        
        # The distance between connections is computed likewise
        conn_distance = dict()  # Dictionary mapping connections (genome0, genome1) to their distance
        
        # Get all the distances of the comparable connections (does not contain duplicates)
        for conn0_id, conn0 in used_conn0.items():
            for conn1_id, conn1 in used_conn1.items():
                if self.node_cache.map_conn_key(conn0_id) == self.node_cache.map_conn_key(conn1_id):
                    conn_distance[conn0_id, conn1_id] = conn0.distance(conn1, cfg=self.config)
        
        # Get all the disjoint connections, these are the connections not present in conn_distance (were not comparable)
        disjoint_connections = 0
        for conn0_id, conn0 in used_conn0.items():
            if not {conn0_id for (a, _) in conn_distance.keys() if a == conn0_id}:
                disjoint_connections += 1
        for conn1_id, conn1 in used_conn1.items():
            if not {conn1_id for (_, b) in conn_distance.keys() if b == conn1_id}:
                disjoint_connections += 1
        
        # The final connection-distance is equal to linear sum of the average weight-difference of the comparable
        #  connections with the number of disjoint/excess connections that are divided by the total number of
        #  connections from the larger gene
        # Note that min(max_conn) == 2, this is a small bias to prevent several "(0,1)" populations!
        max_conn = max(len(used_conn0), len(used_conn1), 2)
        if conn_distance:
            distance += sum(conn_distance.values()) / len(conn_distance)
        distance += (self.config.compatibility_disjoint_conn * disjoint_connections) / max_conn
        
        # Save and return the final distance
        self.distances[g0, g1] = distance
        self.kwargs[g0, g1] = (disjoint_nodes, disjoint_connections)
        return distance
    
    def get_disjoint_genes(self, genome0, genome1):
        """Get the disjoint genes of two given genomes. Return tuple (disjoint_nodes, disjoint_connections)"""
        # Lowest key is always first, removes redundant distance checks
        g0, g1 = (genome0.key, genome1.key) if (genome0.key <= genome1.key) else (genome1.key, genome0.key)
        
        # If key not present, then this means that the distance-measure is not yet determined
        if (g0, g1) not in self.kwargs:
            self(genome0, genome1)
        return self.kwargs[g0, g1]
    
    def warm_up(self, genome_dict: dict):
        """
        Perform a warm-up such that all comparable nodes are merged in the NodeComparingCache. This step must be taken
        in advance to ensure that the minimal distances are computed.
        """
        for genome1_id, genome1 in genome_dict.items():
            for genome2_id, genome2 in genome_dict.items():
                if genome1_id > genome2_id:  # Redundant checks
                    continue
                elif genome1_id == genome2_id:
                    self.distances[genome1_id, genome2_id] = 0
                    continue
                
                # Perform the distance-measuring
                self(genome1, genome2)
        self.distances = dict()
    
    def remove_genome(self, genome):
        """Remove the genome from the cache."""
        keys_to_remove = {(a, b) for (a, b) in self.distances.keys() if (a == genome.key) or (b == genome.key)}
        for k in keys_to_remove:
            self.distances.pop(k)
            self.kwargs.pop(k)
    
    def reset(self):
        """Reset the parameters."""
        self.distances = dict()
        self.node_cache = NodeComparingCache()
