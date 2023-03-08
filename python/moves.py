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

class InputList:
    def __init__(self, movements, time):
        self.movements = movements
        self.time = time

class Placement:
    def __init__(self, inputs, location):
        self.inputs = inputs
        self.location = location

MovementMode = Enum('MovementMode', ['ZeroG', 'ZeroGComplete', 'TwentyG', 'HardDropOnly'])
