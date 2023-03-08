class Row:
    def __init__(self):
        pass


class Board:
    def __init__(self, cells: list[Row], col_heights: list[Row], combo: int,
                  b2b_bonus: bool, hold_piece, next_pieces, bag):
        self.cells = cells
        self.col_heights = col_heights
        self.combo = combo
        self.b2b_bonus = b2b_bonus
        self.hold_piece = hold_piece
        self.next_pieces = next_pieces
        self.bag = bag
