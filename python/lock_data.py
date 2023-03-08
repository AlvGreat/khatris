from enum import Enum

class LockResult:
    def __init__(self, placement_kind, locked_out, b2b, perfect_clear, 
                 combo, garbage_sent, cleared_lines):
        self.placement_kind = placement_kind
        self.locked_out = locked_out
        self.b2b = b2b
        self.perfect_clear = perfect_clear
        self.combo = combo
        self.garbage_sent = garbage_sent
        self.cleared_lines = cleared_lines

PlacementKind = Enum('PlacementKind', ['Clear1', 'Clear2', 'Clear3', 'Clear4', 'MiniTspin', 'MiniTspin1', 
                                        'MiniTspin2', 'Tspin', 'Tspin1', 'Tspin2', 'Tspin3',])
