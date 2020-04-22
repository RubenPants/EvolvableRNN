"""
main_config.py

Configuration file inherited by all the child-configuration files.
"""


class BaseConfig:
    """General __str__ en __repr__ files, enforce classes to have a read() method."""
    
    __slots__ = {}
    
    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return str(self)
    
    def read(self):
        """Convert config-file to a readable format."""
        result = f"{self}:\n"
        for param in sorted(self.__slots__):
            result += f"\t> {param} = {getattr(self, param)}\n"
        return result
    
    def update(self, main_config):
        """Update the parameters"""
        pass
