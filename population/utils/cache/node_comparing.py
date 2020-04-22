"""
genome_comparing.py

Cache to calculate and store the node-distance.
"""
from itertools import count

from configs.genome_config import GenomeConfig
from population.utils.gene_util.output_node import OutputNodeGene


class NodeComparingCache:
    def __init__(self):
        # Holds the boolean indicating if the two nodes are comparable of not
        self.comparable = dict()
        
        # Custom indexer mapping node_keys to internal key-representation
        self.indexer = count(0)
        self.index_map = dict()
    
    def __str__(self):
        result = f"DistanceCache(\n" \
                 f"\tindexer={self.indexer},\n" \
                 f"\tindex_map={self.index_map})"
        return result
    
    def __call__(self, node0, node1, conn0: dict, conn1: dict, cfg: GenomeConfig):
        """
        Define the distance between two nodes. If the two nodes are considered disjoint, no distance is measured.

        :param node0: Node of first genome for which a distance must be determined
        :param node1: Node of second genome for which a distance must be determined
        :param conn0: Connections corresponding the first genome
        :param conn1: Connections corresponding the second genome
        :param cfg: GenomeConfig object

        :return: (False, None) if the given nodes are disjoint
                 (True, dist) otherwise, with dist being the distance between the nodes
        """
        # Get the keys (sorted)
        key = (node0.key, node1.key) if (node0.key <= node1.key) else (node1.key, node0.key)
        
        # Return result if keys already in cache
        if key in self.comparable:
            if self.comparable[key]:
                # Note that node0.distance(node1, cfg) is not stored since this can still vary
                return True, node0.distance(node1, cfg=cfg)
            else:
                return False, None
        
        # Generate new internal key for those that are not yet mapped
        if node0.key not in self.index_map: self.index_map[node0.key] = f"internal_{next(self.indexer)}"
        if node1.key not in self.index_map: self.index_map[node1.key] = f"internal_{next(self.indexer)}"
        
        # If two nodes share the same key, they are always comparable
        if node0.key == node1.key:
            self.comparable[key] = True
            return True, node0.distance(node1, cfg=cfg)
        
        # If both nodes are outputs but do not share the same key, they are not comparable
        if node0.__class__ == node1.__class__ == OutputNodeGene:
            self.comparable[key] = False
            return False, None
        
        # Two nodes are not comparable if they are of different classes
        if node0.__class__ != node1.__class__:
            self.comparable[key] = False
            return False, None
        
        # Two nodes need to have the same incoming connections if they want to be comparable
        inp1 = {a for (a, b) in conn0.keys() if b == node0.key}
        self.map_index(inp1)
        inp2 = {a for (a, b) in conn1.keys() if b == node1.key}
        self.map_index(inp2)
        if inp1 != inp2:
            # Save and return the result
            self.comparable[key] = False
            return False, None
        
        # The two keys represent the same node, merge their internal representation
        self.merge_keys(key1=node0.key, key2=node1.key)
        
        # The two nodes can be compared, define their distance
        self.comparable[key] = True
        d = node0.distance(node1, cfg=cfg)
        return True, d
    
    def map_index(self, inp: set):
        """Map the inputs-set's keys towards the internal representation."""
        for i in inp.copy():
            if i in self.index_map:
                inp.remove(i)
                inp.add(self.index_map[i])
    
    def map_conn_key(self, key: tuple):
        """Map a tuple of two integers, representing a connecting gene's key, to the internal representation."""
        k0 = self.index_map[key[0]] if key[0] in self.index_map else key[0]
        k1 = self.index_map[key[1]] if key[1] in self.index_map else key[1]
        return k0, k1
    
    def merge_keys(self, key1, key2):
        """Merge two indexer-keys together."""
        assert key1 in self.index_map
        assert key2 in self.index_map
        
        # Get the internal representations
        v1, v2 = self.index_map[key1], self.index_map[key2]
        
        # Convert all v2 occurrences into v1's
        for k, v in self.index_map.items():
            # Share the same internal representation
            if v == v2:
                self.index_map[k] = v1
            
                # Given node is now comparable with both the merged nodes
                self.comparable[(key1, k) if (key1 <= k) else (k, key1)] = True
                self.comparable[(key2, k) if (key2 <= k) else (k, key2)] = True
    
    def reset(self):
        """Reset the cache's parameters."""
        self.comparable = dict()
        self.indexer = count(0)
        self.index_map = dict()
