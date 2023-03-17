use pyo3::{prelude::*, pyclass};
use libtetris::*;
use enumset::EnumSet;

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
    fn new(inputs: Vec<String>, time: u32, piece_type: String, rotation_state: String, x:i32, y:i32, tspin:String) -> Self {
        PiecePlacement {inputs, time, piece_type, rotation_state, x, y, tspin }
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

    let placements = find_moves(&game_board, piece, match mode {
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
fn get_placement_res(board_arr: [[bool; 10]; 40], hold: char, bag_remain: [char; 6], b2b: bool, 
                     combo: u32, piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8) -> PyResult<()> {
    let converted_hold: Option<Piece> = match hold {
        ' ' => None,
        _ => Some(piece_str_to_enum(hold))
    };

    // Convert a list of pieces in string format to an EnumSet of Piece enums
    let mut piece_set = EnumSet::new();
    let converted_pieces: Vec<Piece> = bag_remain.iter().map(|x| piece_str_to_enum(*x)).collect();
    for p in converted_pieces {
        piece_set.insert(p);
    }

    // Initialize board with existing stats and place the new piece in
    let mut game_board: Board = Board::new_with_state(board_arr, piece_set, converted_hold, b2b, combo);
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

    let _lock_res: LockResult = game_board.lock_piece(placed_piece);

    // Need to convert the results back into format for Python

    Ok(())
}


/// A Python module implemented in Rust.
#[pymodule]
fn pylibtetris(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_moves_py, m)?)?;
    Ok(())
}
