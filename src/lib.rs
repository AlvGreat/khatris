use pyo3::{prelude::*, pyclass};
use libtetris::*;
use enumset::EnumSet;

#[pyclass]
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


#[pyfunction]
fn find_moves_py(board: [[bool; 10]; 40], piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8, mode: u8) -> PyResult<Vec<PiecePlacement>> {
    // this initializes the game board using a boolean array
    let game_board: Board = Board::new_with_state(board, EnumSet::all(), None, false, 0);
    let piece = FallingPiece {
        kind: PieceState(match piece {
            'I' => Piece::I,
            'O' => Piece::O,
            'T' => Piece::T,
            'L' => Piece::L,
            'J' => Piece::J,
            'S' => Piece::S,
            'Z' => Piece::Z,
            _ => Piece::O,
        }, match rotation_state {
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
        let py_placement = PiecePlacement::new(input_vec, placements[i].inputs.time, placements[i].location.kind.0.to_string(), placements[i].location.kind.1.to_string(), placements[i].location.x, placements[i].location.y, placements[i].location.tspin.to_string());
        py_placements.push(py_placement);
    }
    Ok(py_placements)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pylibtetris(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_moves_py, m)?)?;
    Ok(())
}
