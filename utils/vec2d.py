"""
vec2d.pyx

Representation for two-dimensional points.
"""
import operator
from math import atan2, cos, sin, sqrt


class Vec2d(object):
    """Create a two dimensional vector (or point in space)."""
    
    __slots__ = {
        "x", "y"
    }
    
    def __init__(self, x: float = 0, y: float = 0):
        self.x: float = x
        self.y: float = y
    
    def __str__(self):
        return f"Vec2d({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash(str(self))
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError()
    
    def __setitem__(self, i: int, value: float):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __len__(self):
        return 2
    
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False
    
    def __lt__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            if self.x < other[0]:
                return True
            elif (self.x == other[0]) and (self.y < other[1]):
                return True
            else:
                return False
        else:
            raise TypeError(f"Not possible to use '<' operator on objects {type(self)} and {type(other)}")
    
    def __gt__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            if self.x > other[0]:
                return True
            elif (self.x == other[0]) and (self.y > other[1]):
                return True
            else:
                return False
        else:
            raise TypeError(f"Not possible to use '>' operator on objects {type(self)} and {type(other)}")
    
    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True
    
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)
    
    __radd__ = __add__
    
    def __iadd__(self, other):
        if isinstance(other, Vec2d):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Vec2d(other[0] - self.x, other[1] - self.y)
        else:
            return Vec2d(other - self.x, other - self.y)
    
    def __isub__(self, other):
        if isinstance(other, Vec2d):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self
    
    def __mul__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x * other.x, self.y * other.y)
        if hasattr(other, "__getitem__"):
            return Vec2d(self.x * other[0], self.y * other[1])
        else:
            return Vec2d(self.x * other, self.y * other)
    
    __rmul__ = __mul__
    
    def __imul__(self, other):
        if isinstance(other, Vec2d):
            self.x *= other.x
            self.y *= other.y
        elif hasattr(other, "__getitem__"):
            self.x *= other[0]
            self.y *= other[1]
        else:
            self.x *= other
            self.y *= other
        return self
    
    def _operator_handler(self, other, f):
        if isinstance(other, Vec2d):
            return Vec2d(f(self.x, other.x),
                         f(self.y, other.y))
        elif hasattr(other, "__getitem__"):
            return Vec2d(f(self.x, other[0]),
                         f(self.y, other[1]))
        else:
            return Vec2d(f(self.x, other),
                         f(self.y, other))
    
    def _right_operator_handler(self, other, f):
        if hasattr(other, "__getitem__"):
            return Vec2d(f(other[0], self.x),
                         f(other[1], self.y))
        else:
            return Vec2d(f(other, self.x),
                         f(other, self.y))
    
    def _inplace_operator_handler(self, other, f):
        if hasattr(other, "__getitem__"):
            self.x = f(self.x, other[0])
            self.y = f(self.y, other[1])
        else:
            self.x = f(self.x, other)
            self.y = f(self.y, other)
        return self
    
    def __div__(self, other):
        return self._operator_handler(other, operator.div)
    
    def __rdiv__(self, other):
        return self._right_operator_handler(other, operator.div)
    
    def __idiv__(self, other):
        return self._inplace_operator_handler(other, operator.div)
    
    def __floordiv__(self, other):
        return self._operator_handler(other, operator.floordiv)
    
    def __rfloordiv__(self, other):
        return self._right_operator_handler(other, operator.floordiv)
    
    def __ifloordiv__(self, other):
        return self._inplace_operator_handler(other, operator.floordiv)
    
    def __truediv__(self, other):
        return self._operator_handler(other, operator.truediv)
    
    def __rtruediv__(self, other):
        return self._right_operator_handler(other, operator.truediv)
    
    def __itruediv__(self, other):
        return self._inplace_operator_handler(other, operator.truediv)
    
    def __neg__(self):
        return Vec2d(operator.neg(self.x), operator.neg(self.y))
    
    def __pos__(self):
        return Vec2d(operator.pos(self.x), operator.pos(self.y))
    
    def __abs__(self):
        return Vec2d(abs(self.x), abs(self.y))
    
    def __invert__(self):
        return Vec2d(-self.x, -self.y)
    
    def __round__(self, n=0):
        return Vec2d(round(self.x, n), round(self.y, n))
    
    def __copy__(self):
        return Vec2d(self.x, self.y)
    
    def get_angle(self):
        return 0 if self.get_length() == 0 else atan2(self.y, self.x)
    
    def get_length(self):
        return sqrt(self.x ** 2 + self.y ** 2)
    
    def normalized(self):
        return self / self.get_length() if self.get_length() != 0 else Vec2d(self)
    
    def get_tuple(self, r: int = 0):
        if r:
            return round(self.x, r), round(self.y, r)
        return self.x, self.y
    
    def load_tuple(self, t: tuple):
        """Load in a tuple to the object's parameters."""
        self.x = t[0]
        self.y = t[1]
        return self


def angle_to_vec(angle: float):
    """
    Transform an angle to a normalized vector.

    :param angle: Float
    :return: Vec2d
    """
    return Vec2d(cos(angle), sin(angle))
