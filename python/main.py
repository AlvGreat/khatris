from pylibtetris.pylibtetris import *

#print(pylibtetris.__)
# help(pylibtetris)
#print(pylibtetris.__dir__())

# y = PyLockResult(1, True, True, 0, 1)
# x = PiecePlacement(["CCW"], 1, "T", "North", 1, 1, "T")

# print(x)
# print(y)

blank_board = [[False for _ in range(10)] for _ in range(40)]

test_board = [4*[True] + 6*[False]]
test_board += [[False for _ in range(10)] for _ in range(39)]

test_pyboard = PyBoard(test_board, "T", True, 5, ["S", "Z", "L", "J", "I", "O"], ["S", "Z", "L", "J", "I", "O", "T"])
print(test_pyboard)

# fn find_moves_py(board: [[bool; 10]; 40], piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8, mode: u8)

piece_placements = find_moves_py(test_board, 'S', 0, 5, 19, 0, 0)
print(piece_placements[0])

# fn get_placement_res(board_arr: [[bool; 10]; 40], hold: char, b2b: bool, combo: u32, next_pieces: [char; 6], bag: [char; 7],
#                      piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8) -> PyResult<(PyBoard, PyLockResult)> {

new_board, lock_res = get_placement_res(test_pyboard, 'S', 1, 8, 1, 0)
print(new_board)
