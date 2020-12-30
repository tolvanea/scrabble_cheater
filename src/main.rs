mod board;
mod my_prelude;

use my_prelude::*;
use board::Board;
use rand::Rng;

pub fn metropolis_solve(mut board: Board, temp: f64, blocks: usize) -> (Board, usize, f64) {
    let mut fitness_old_state = board.fitness();
    for b in 0..blocks {
        let old_table = board.table.clone();
        let old_placed = board.placed.clone();
        board.random_move();
        let fitness_new_state = board.fitness();
        if fitness_new_state < 0.0 {
            println!("Solution found!");
            board.fitness2();
            return (board, b, fitness_new_state);
        }
        let diff = fitness_new_state - fitness_old_state;
        let rnd = board.rng.gen_range(0.0..1.0);
        if (-diff / temp).exp() > rnd {
            fitness_old_state = fitness_new_state;
        } else {
            board.table = old_table;
            board.placed = old_placed;
        }
    }
    return (board, blocks, fitness_old_state);
}

/*
1e0
Fitness 5737,              Iters: 500000, Elapsed: 5751 ms
Fitness 5768.803847577294, Iters: 500000, Elapsed: 5972 ms
Fitness 5737,              Iters: 500000, Elapsed: 5751 ms

3e0
Fitness 5817.975420452547, Iters: 500000, Elapsed: 6206 ms
Fitness 5811.514718625762, Iters: 500000, Elapsed: 6381 ms
Fitness 5816.146993327801, Iters: 500000, Elapsed: 6106 ms

3e-1
Fitness 5782.975420452547, Iters: 500000, Elapsed: 5807 ms
Fitness 5778,              Iters: 500000, Elapsed: 5946 ms
Fitness 5780.171572875254, Iters: 500000, Elapsed: 5882 ms




 */

fn run_once(letters: Vec<char>) {
    let tot_iters = 500_000;
    let parts = 10;
    let mut board = Board::new(16, letters);
    let mut iterated = 0;
    board.draw();
    let mut fit = 0.0;
    let before = std::time::Instant::now();
    for _ in 0..parts {
        let out = metropolis_solve(board.clone(), 1e0, tot_iters/parts);
        board = out.0;
        iterated += out.1;
        fit = out.2;
        println!("Fitness {}", fit);
        board.draw();
        if fit < 0.0 {
            break;
        }
    }
    println!("Fitness {}, Iters: {}, Elapsed: {} ms", fit, iterated,  before.elapsed().as_millis())
}

#[allow(dead_code)]
fn benchmark() {
    let letters: Vec<char> = "aaeiiouykklmnrtpsv".chars().collect();
    for &temp in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2].iter() {
        let num_samples = 10;
        let mut samples= nd::Array1::from_elem(num_samples, 0.0);
        let mut board = Board::new(16, letters.clone());
        for fit in samples.iter_mut() {
            let (_board_sol, _iters, _) = metropolis_solve(board.clone(), temp, 100);
            *fit = board.fitness();
            board.initial_position();
        }
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let std = samples.std_axis(Axis(0), 0.0) / (samples.len() as f64).sqrt();
        println!("Temp: {:e}, fitness: {}({})", temp, mean, std);
    }
}




/// Extract UTF-8 compatible string slice by using character indices.
/// If given indices are outside character boundaries, then valid part of string slice is
/// returned in `Err`-variant.
///
/// If string slice contains of only ascii characters, then this function is
/// equivalent to `&s[begin..(begin+length)]`.
///
/// # Examples
/// ```
/// let s = "abc🙂";
/// assert_eq!(Ok("bc"), substr(s, 1, Some(2)));
/// assert_eq!(Ok("c🙂"), substr(s, 2, Some(2)));
/// assert_eq!(Ok("c🙂"), substr(s, 2, None));
/// assert_eq!(Err("c🙂"), substr(s, 2, Some(99)));
/// assert_eq!(Ok(""), substr(s, 2, Some(0)));
/// assert_eq!(Ok(""), substr(s, 4, Some(0)));
/// assert_eq!(Err(""), substr(s, 5, Some(0)));
/// assert_eq!(Err(""), substr(s, 5, Some(4)));
/// assert_eq!(Ok(""), substr("", 0, Some(0)));
/// assert_eq!(Ok(""), substr("", 0, Some(0)));
/// assert_eq!(Ok("y̆"), substr("y̆es", 0, Some(1)));
/// assert_eq!(Ok("es"), substr("y̆es", 1, Some(2)));
/// ```
/// TODO add negative indices, publish a crate
#[allow(dead_code)]
fn substr(s: &str, begin: usize, length: Option<usize>) -> Result<&str, &str> {
    use std::iter::once;
    use unicode_segmentation::UnicodeSegmentation;
    let mut itr = s.grapheme_indices(true).map(|(n, _)| n).chain(once(s.len()));
    let beg = itr.nth(begin);
    if beg.is_none() {
        return Err("");
    } else if length == Some(0) {
        return Ok("");
    }
    let end = length.map_or(Some(s.len()), |l| itr.nth(l-1));
    if let Some(end) = end {
        return Ok(&s[beg.unwrap()..end]);
    } else {
        return Err(&s[beg.unwrap()..s.len()]);
    }
}

fn main() -> Res<()> {
    let letters = "aaeiiouykklmnrtpsv".chars().collect();
    run_once(letters);
    //benchmark();
    //board::stress_test();
    return Ok(());
}
