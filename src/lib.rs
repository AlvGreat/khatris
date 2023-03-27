use pyo3::{prelude::*, pyclass};
use libtetris::*;
use enumset::EnumSet;
use std::collections::VecDeque;

#[pyclass]
#[derive(Clone)]
struct PiecePlacement {
    #[pyo3(get, set)]
    inputs: Vec<String>,
    #[pyo3(get, set)]
    time: u32,
    #[pyo3(get, set)]
    piece_type: String,
    #[pyo3(get, set)]
    rotation_state: String,
    #[pyo3(get, set)]
    x: i32,
    #[pyo3(get, set)]
    y: i32,
    #[pyo3(get, set)]
    tspin: String,
}

#[pymethods]
impl PiecePlacement {
    #[new]
    fn new(inputs: Vec<String>, time: u32, piece_type: String, rotation_state: String, x: i32, y: i32, tspin: String) -> Self {
        PiecePlacement { inputs, time, piece_type, rotation_state, x, y, tspin }
    }

    fn __str__(&self) -> String {
        format!("PiecePlacement\n  Inputs: {} \n  Time: {} \n  Piece Type: {} \n  Rotation: {} \n  (x, y): ({}, {}) \n  Tspin: {}", 
                self.inputs.join(", "),
                self.time.to_string(),
                self.piece_type,
                self.rotation_state,
                self.x.to_string(),
                self.y.to_string(),
                self.tspin)
    }
}


#[pyclass]
#[derive(Clone)]
struct PyBoard {
    #[pyo3(get, set)]
    field: [[bool; 10]; 40],
    #[pyo3(get, set)]
    hold: char, 
    #[pyo3(get, set)]
    b2b: bool, 
    #[pyo3(get, set)]
    combo: u32,
    #[pyo3(get, set)]
    next_pieces: [char; 6], 
    #[pyo3(get, set)]
    bag: [char; 7] 
}

#[pymethods]
impl PyBoard {   
    #[new]
    fn new(field: [[bool; 10]; 40], hold: char, b2b: bool, combo: u32, next_pieces: [char; 6], bag: [char; 7]) -> Self {
        PyBoard { field, hold, b2b, combo, next_pieces, bag }
    }

    fn __str__(&self) -> String {
        let mut board_str = String::new();
        
        // go from row 0 to 20 but print in reverse order (the board is upside down)
        for i in (0..20).rev() {
            board_str.push('\t');
            for j in 0..self.field[i].len() {
                if self.field[i][j] {
                    board_str.push('x');
                } else {
                    board_str.push('o');
                }
            }
            if i < self.field.len() - 1 {
                board_str.push('\n');
            }
        }

        format!("PyBoard\n  Field:\n{} \n  Hold: {} \n  Queue: {} \n  B2B: {} \n  Combo: {} \n  Bag: {}", 
                board_str,
                self.hold.to_string(),
                self.next_pieces.iter().map(|c| c.to_string()).collect::<Vec<String>>().join(", "),
                self.b2b.to_string(),
                self.combo.to_string(),
                self.bag.iter().map(|c| c.to_string()).collect::<Vec<String>>().join(", "))
    }
}

#[pyfunction]
fn new_board_with_queue() -> PyBoard {
    let mut board = Board::new();
    for _ in 0..6 {
        board.add_next_piece(board.generate_next_piece(&mut rand::thread_rng()));
    }
    new_pyboard(board)
}

fn new_pyboard(mut board: Board) -> PyBoard {
    let mut bag = [' '; 7];
    let mut i = 0;
    for piece in board.bag.iter() {
        bag[i] = piece_enum_to_str(piece);
        i += 1;
    }

    return PyBoard {
        field: board.get_field(), 
        hold: piece_opt_enum_to_str(board.hold_piece),
        b2b: board.b2b_bonus,
        combo: board.combo,
        next_pieces: board.get_queue_arr(),
        bag: bag
    };
}


#[pyclass]
struct PyLockResult {
    #[pyo3(get, set)]
    placement_kind: u32,
    #[pyo3(get, set)]
    b2b: bool,
    #[pyo3(get, set)]
    perfect_clear: bool,
    #[pyo3(get, set)]
    combo: u32,
    #[pyo3(get)]
    garbage_sent: u32
}

#[pymethods]
impl PyLockResult {
    #[new]
    fn new(placement_kind: u32, b2b: bool, perfect_clear: bool, combo: u32, garbage_sent: u32) -> Self {
        PyLockResult { placement_kind, b2b, perfect_clear, combo, garbage_sent }
    }

    fn __str__(&self) -> String {
        format!("PyLockResult\n  Placement Kind: {} \n  B2B: {} \n  Perfect Clear: {} \n  Combo: {} \n  Garbage Sent: {}", 
                self.placement_kind,
                self.b2b,
                self.perfect_clear,
                self.combo,
                self.garbage_sent)
    }
}

fn new_pylockresult(lock_res: LockResult) -> PyLockResult {
    PyLockResult {
        placement_kind: match lock_res.placement_kind {
            PlacementKind::None => 0,
            PlacementKind::Clear1 => 1,
            PlacementKind::Clear2 => 2,
            PlacementKind::Clear3 => 3,
            PlacementKind::Clear4 => 4,
            PlacementKind::MiniTspin => 5,
            PlacementKind::MiniTspin1 => 6,
            PlacementKind::MiniTspin2 => 7,
            PlacementKind::Tspin => 8,
            PlacementKind::Tspin1 => 9,
            PlacementKind::Tspin2 => 10,
            PlacementKind::Tspin3 => 11
        },
        b2b: lock_res.b2b,
        perfect_clear: lock_res.perfect_clear,
        combo: lock_res.combo.unwrap_or(0),
        garbage_sent: lock_res.garbage_sent
    }
}

fn piece_str_to_enum(piece: char) -> Piece {
    match piece {
        'I' => Piece::I,
        'O' => Piece::O,
        'T' => Piece::T,
        'L' => Piece::L,
        'J' => Piece::J,
        'S' => Piece::S,
        'Z' => Piece::Z,
        _ => Piece::O,
    }
}

fn piece_enum_to_str(piece: Piece) -> char {
    match piece {
        Piece::I => 'I',
        Piece::O => 'O',
        Piece::T => 'T',
        Piece::L => 'L',
        Piece::J => 'J',
        Piece::S => 'S',
        Piece::Z => 'Z'
    }
}

fn piece_opt_enum_to_str(piece: Option<Piece>) -> char {
    match piece {
        Some(Piece::I) => 'I',
        Some(Piece::O) => 'O',
        Some(Piece::T) => 'T',
        Some(Piece::L) => 'L',
        Some(Piece::J) => 'J',
        Some(Piece::S) => 'S',
        Some(Piece::Z) => 'Z',
        None => ' '
    }
}

#[pyfunction]
fn find_moves_py(board: [[bool; 10]; 40], piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8, mode: u8) -> PyResult<Vec<PiecePlacement>> {
    // initialize board with boolean array and existing stats
    let game_board: Board = Board::new_with_state(board, EnumSet::all(), None, false, 0);
    let piece = FallingPiece {
        kind: PieceState(piece_str_to_enum(piece), match rotation_state {
            0 => RotationState::North,
            1 => RotationState::South,
            2 => RotationState::East,
            3 => RotationState::West,
            _ => RotationState::North,
        }),
        x: x,
        y: y,
        tspin: match t_spin_status {
            0 => TspinStatus::None,
            1 => TspinStatus::Mini,
            2 => TspinStatus::Full,
            _ => TspinStatus::None,
        },
    };  

    let placements:Vec<Placement> = find_moves(&game_board, piece, match mode {
        0 => MovementMode::ZeroG,
        1 => MovementMode::ZeroGComplete,
        2 => MovementMode::TwentyG,
        3 => MovementMode::HardDropOnly,
        _ => MovementMode::ZeroG,
    });

    let mut py_placements = Vec::new();
    
    for i in 0..placements.len() {
        let mut input_vec: Vec<String> = Vec::new();
        for j in 0..placements[i].inputs.movements.len() {
            input_vec.push(placements[i].inputs.movements[j].to_string());
        }
        let py_placement = PiecePlacement::new(input_vec, 
            placements[i].inputs.time, 
            placements[i].location.kind.0.to_string(), 
            placements[i].location.kind.1.to_string(), 
            placements[i].location.x, 
            placements[i].location.y, 
            placements[i].location.tspin.to_string()
        );
        py_placements.push(py_placement);
    }
    Ok(py_placements)
}


// Accept the board and PiecePlacement data
// Return information about what happens after the input piece is placed
#[pyfunction]
fn get_placement_res(py_board: PyBoard, piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status: i8) -> PyResult<(PyBoard, PyLockResult)> {
    let converted_hold: Option<Piece> = match py_board.hold {
        ' ' => None,
        _ => Some(piece_str_to_enum(py_board.hold))
    };

    // Convert a list of pieces in string format to an EnumSet of Piece enums for the bag
    let mut bag_set = EnumSet::new();
    let converted_pieces: Vec<Piece> = py_board.bag.iter().map(|x| piece_str_to_enum(*x)).collect();
    for p in converted_pieces {
        bag_set.insert(p);
    }

    // Create a Deque from the next_pieces array for the Tetris board queue
    // Note that we need to convert each character into the Piece enum
    let mut piece_queue: VecDeque<Piece> = VecDeque::new();
    let converted_pieces: Vec<Piece> = py_board.next_pieces.iter().map(|x| piece_str_to_enum(*x)).collect();
    for p in converted_pieces {
        piece_queue.push_back(p);
    }

    // Initialize board with existing stats and place the new piece in
    let mut game_board: Board = Board::new_with_state(py_board.field, bag_set, converted_hold, py_board.b2b, py_board.combo);
    game_board.set_queue(piece_queue);

    // Now, we generate a new piece, add it to the queue, and then advance the queue
    let new_piece = game_board.generate_next_piece(&mut rand::thread_rng());
    game_board.add_next_piece(new_piece);
    game_board.advance_queue();

    let placed_piece = FallingPiece {
        kind: PieceState(piece_str_to_enum(piece), match rotation_state {
            0 => RotationState::North,
            1 => RotationState::South,
            2 => RotationState::East,
            3 => RotationState::West,
            _ => RotationState::North,
        }),
        x: x,
        y: y,
        tspin: match t_spin_status {
            0 => TspinStatus::None,
            1 => TspinStatus::Mini,
            2 => TspinStatus::Full,
            _ => TspinStatus::None,
        },
    };

    let lock_res: LockResult = game_board.lock_piece(placed_piece);
    
    // Convert the results back into format for Python
    Ok((new_pyboard(game_board), new_pylockresult(lock_res)))
}


/// A Python module implemented in Rust.
#[pymodule]
fn pylibtetris(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_moves_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_placement_res, m)?)?;
    m.add_function(wrap_pyfunction!(new_board_with_queue, m)?)?;
    m.add_class::<PiecePlacement>()?;
    m.add_class::<PyBoard>()?;
    m.add_class::<PyLockResult>()?;
    Ok(())
}
