#![allow(clippy::needless_return)]
mod board;
mod my_prelude;

use my_prelude::*;
use board::Board;

pub fn metropolis(mut board: Board, temp: f64, blocks: usize) -> (Board, usize, f64, usize, bool) {
    let mut fitness_old = board.fitness;
    for block in 0..blocks {
        let table_old = board.table.clone();
        let placed_old = board.placed.clone();
        let priorities_old = board.priorities.clone();
        board.random_move();
        let fitness_new = board.fitness;
        if fitness_new.2 {
            return (board, block, fitness_new.0, fitness_new.1, true);
        }
        let diff = fitness_new.0 - fitness_old.0;
        if (-diff / temp).exp() > board.random_float_0_1() {
            fitness_old = fitness_new;
        } else {
            board.table = table_old;
            board.placed = placed_old;
            board.priorities = priorities_old;
        }
    }
    return (board, blocks, fitness_old.0, fitness_old.1, false);
}


#[allow(dead_code)]
fn run_once(given_letters: Vec<char>) {
    let tot_iters = 1_000_000;
    let parts = 100;
    let constants = Some([20, 10, 5, 50, 30]);
    let mut board = Board::new(20, given_letters.clone(), constants);
    let mut iterated = 0;
    let mut letters = 0;
    let mut fitness = 0.0;
    let mut sol  = false;
    board.draw();
    let before = std::time::Instant::now();
    for _ in 0..parts {
        let out = metropolis(board.clone(), 3.0, tot_iters/parts);
        board = out.0;
        iterated += out.1;
        fitness = out.2;
        letters = out.3;
        board.draw();
        if out.4 {
            sol = true;
            break;
        }
        println!("Fitness {}", fitness);
    }
    if sol {
        println!("Solution found!");
    } else {
        println!("Solution not found :(");
    }
    println!(
        "Correct letters: {}/{}, Elapsed: {} ms, Iters: {}, Fitness {:.2}, ",
        letters, given_letters.len(), before.elapsed().as_millis(), iterated, fitness,
    )
}

/// This function can be used to search good set of parameters. Parallel threads are used.
#[allow(dead_code)]
fn find_good_parameters() {
    use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};
    //use rayon::prelude::*;
    let params = [
        (4.0, Some([20, 10, 5, 50, 30])),
        (4.0, Some([20, 10, 7, 50, 30])),
        (4.0, Some([20, 10, 3, 50, 30])),
        (4.0, Some([20, 10, 5, 50, 20])),
        (4.0, Some([20, 10, 5, 50, 40])),
        (3.5, Some([20, 10, 5, 50, 30])),
        (4.5, Some([20, 10, 5, 50, 30])),
    ];

    let letters: Vec<char> = "aaeiioukklmnrtsv".chars().collect();

    let board = Board::new(16, letters, None);
    let mean_std: Vec<_> = params.par_iter().map(|(temp, c)| {
        let num_samples = 100;
        let mut board = board.clone();
        board.constants = c.unwrap();
        let mut samples= nd::Array2::from_elem((num_samples, 3), 0.0);
        for mut sample in samples.axis_iter_mut(Axis(0)) {
            let (_sol, iters, fitness, letters, _) = metropolis(board.clone(), *temp, 100_000);
            sample[0] = fitness;
            sample[1] = letters as f64;
            sample[2] = iters as f64;
            board.initialize();
        }
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let std = samples.std_axis(Axis(0), 0.0) / (num_samples as f64).sqrt();
        (mean, std)
    }).collect();

    println!("{:>39} {:<14}  {:<10}  {:<10}", "", "fitness", "letters", "iters");
    for ((mean, std), (temp, c)) in mean_std.iter().zip(params.iter()) {
        println!(
            "Temp {:.1}, params {:?}:  {:.2}({:.2}),  {:.2}({:.2}), {:.2}({:.2})",
            temp, c.unwrap(), mean[0], std[0], mean[1], std[1], mean[2], std[2]
        );
    }
}



// This function does not belong here in this project, as it is not anymore used. However,
// it is a good general algorithm to slice strings.
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
    let given_letters = "tnahivonajsnäenoilsteailk".chars().collect();
    run_once(given_letters);
    //find_good_parameters();
    return Ok(());
}
