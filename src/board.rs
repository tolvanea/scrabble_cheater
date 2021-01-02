use crate::my_prelude::*;
use std::collections::HashSet;
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;
pub use rand::{thread_rng, seq::SliceRandom, Rng, rngs::StdRng, SeedableRng};
pub use rand::distributions::{Distribution, Uniform};

use petgraph::{graph::UnGraph, algo::tarjan_scc};
use std::convert::From;

const MAX_WORD_LENGTH: usize = 16;

// TODO If I ever continue this project, make sure to fix that ugly hack that letter reserve
//      area is first two rows of the table. There should be two vectors `placed` and `unplaced`.
//      (Way to much logic is spent on wiggling this current hacky arrangement.)

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tile {
    /// Single letter tile. Idx is position in `placed` vec
    Occupied { idx: usize, letter: char },
    Empty,
}

impl Tile {
    pub fn unwrap(self) -> (usize, char) {
        match self {
            Tile::Occupied{ idx, letter } => (idx, letter),
            Tile::Empty => panic!("Unwrap failed"),
        }
    }
    pub fn is_occupied(&self) -> bool {
        !matches!(self, Tile::Empty)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TilePriority {
    Word,
    PartialWord,
    Crap,
    NotUsed,
}

#[derive(Debug, Clone)]
pub struct Board {
    /// Board/table that is square grid
    pub table: Array2<Tile>,
    /// All given letters
    pub letters: Vec<char>,
    /// Indices of letters that are placed on board/table
    pub placed: Vec<(usize, usize)>,
    /// Determine if tile is free to use or more important
    pub priorities: Vec<TilePriority>,
    /// Sampling constants
    /// * full word coefficient / 10
    /// * partial word coefficient / 10
    /// * full word letter choice probability-%
    /// * partial word letter choice probability-%
    /// * letter reserve move probability-%
    pub constants: [usize; 5],
    /// Fitness and number of letters that are part of full word
    pub fitness: (f64, usize, bool),
    /// With dictionaries one can test whether string is a real word or part of a real word.
    /// Element [0] is dictionary for full words, element [1] is one for partial words with
    /// lengths 2-16 parts with length 2 and so on.
    pub dictionaries: [HashSet<String>; 2],
    /// Random number generator
    pub rng: (StdRng, Uniform<usize>, Uniform<usize>, Uniform<usize>, Uniform<f64>),
}

// Public impls
impl Board {
    pub fn new(side_length: usize, letters: Vec<char>, constants: Option<[usize; 5]>) -> Board {
        assert!(
            (letters.len() < side_length * 2) && (side_length >= 5) && (letters.len() >= 2),
            "Sorry, `side_length` must be at least half of the letter count. \n\
            It is for board reservation of unused letter.",
        );
        let random_seed = thread_rng().gen::<u64>(); // random seed is used by default
        let rng = (
            SeedableRng::seed_from_u64(random_seed),
            Uniform::from(0..100),
            Uniform::from(0..letters.len()),
            Uniform::from(0..4),
            Uniform::from(0.0..1.0),
        );

        let path = "dictionaries.serde";
        let dictionaries = load_or_generate(path, Self::dicts_for_word_parts).unwrap();
        let table = Array2::from_elem((side_length, side_length), Tile::Empty);
        // self.letters corresponds indices with self.placed
        let placed = Vec::new();
        let priorities = Vec::new();
        let constants = constants.unwrap_or([10, 10, 5, 50, 10]);
        let fitness = (f64::NAN, 0, false);

        let mut board = Board{
            table,
            letters,
            placed,
            priorities,
            constants,
            fitness,
            dictionaries,
            rng,
        };
        board.initialize();
        board.calc_fitness();
        board.invariant().expect("Invarant broken");
        return board;
    }

    pub fn initialize(&mut self) {
        self.placed.clear();
        self.table.iter_mut().for_each(|tile| *tile = Tile::Empty);
        self.priorities = vec![TilePriority::NotUsed; self.letters.len()];

        let mut unplaced = self.letters.clone(); // Unplaced letters
        unplaced.reverse();
        for (id_y, mut row) in self.table.outer_iter_mut().enumerate().take(2) {
            for (id_x, tile) in row.iter_mut().enumerate() {
                match unplaced.pop() {
                    Some(l) => {
                        *tile = Tile::Occupied { idx: self.placed.len(), letter: l };
                        self.placed.push((id_y, id_x));
                    },
                    None => break,
                }
            }
        }
        // Move one tile in the center of board
        let id = self.random_index();
        let new_pos = (self.table.shape()[0] / 2 + 1, self.table.shape()[0] / 2);
        let old_pos = &mut self.placed[id];
        self.table[new_pos] = self.table[*old_pos];
        self.table[*old_pos] = Tile::Empty;
        *old_pos = new_pos;
        self.priorities[id] = TilePriority::Crap;
    }

    pub fn random_move(&mut self) {
        if self.random_presentage() == 0 && self.centerize() {
            return;
        }
        self.invariant().unwrap();

        let side = self.table.shape()[0] as isize;
        let c = self.constants;
        let one_tile_on_table = self.placed.iter().filter(|(y,_)| *y > 2).count() == 1;

        // Loop until some good tile is found that can be moved
        let (id, tile, old_pos) = loop {
            let id = self.random_index();
            match self.priorities[id] {
                TilePriority::Word => if self.random_presentage() >= c[2] { continue },
                TilePriority::PartialWord => if self.random_presentage() >= c[3] {continue },
                TilePriority::Crap | TilePriority::NotUsed => (),
            }
            let old_pos: (usize, usize) = self.placed[id];
            if old_pos.0 >= 2 && one_tile_on_table {
                continue // Nope, do not remove last tile on table, try some other.
            }
            let tile = self.table[old_pos];
            debug_assert!(tile != Tile::Empty);
            self.table[old_pos] = Tile::Empty;
            break (id, tile, old_pos)
        };

        // Loop until new position has been found
        'outer: loop {
            let (mut ni, d) = if (old_pos.0 < 2) || (self.random_presentage() < c[4]) {
                // Trial position of new tile that is at position of existing tile
                let nu = loop {
                    let idx = self.random_index();
                    let nu = self.placed[idx];
                    if nu.0 >= 2 {
                        break nu
                    }
                };
                let ni = (nu.0 as isize, nu.1 as isize);
                // Direction in which new tiles will be searched
                let d = [(-1, 0), (1, 0), (0, 1), (0, -1)][self.random_on_range_0_4()];
                (ni, d)
            } else {
                let y = (self.random_on_range_0_4() / 2) as isize;
                ((y, 0), (0, 1))
            };
            // Step in chosen direction until free tile is found
            loop {
                ni = (ni.0 + d.0, ni.1 + d.1);
                let nu = (ni.0 as usize, ni.1 as usize);
                // Check if new position is valid. First 2 rows are reserved for unplaced letters.
                if (0 <= ni.0 && 0 <= ni.1) && (ni.0 < side && ni.1 < side) && nu != old_pos {
                    if !self.table[nu].is_occupied() {
                        self.placed[id] = nu;
                        self.table[nu] = tile;
                        self.priorities[id] = if ni.0 < 2 {
                            TilePriority::NotUsed
                        } else {
                            TilePriority::Crap
                        };
                        break 'outer;
                    }
                } else {
                    // Random choice was bad. Choose a new one.
                    continue 'outer;
                }
            }
        }
        self.calc_fitness();
    }

    pub fn draw(&self) {
        println!();
        for row in self.table.outer_iter() {
            for square in row.iter() {
                match *square {
                    Tile::Occupied{ idx:_, letter} => print!("{} ", letter),
                    Tile::Empty => print!(". "),
                }
            }
            println!();
        }
    }

    pub fn random_presentage(&mut self) -> usize {
        self.rng.1.sample(&mut self.rng.0)
    }

    pub fn random_index(&mut self) -> usize {
        self.rng.2.sample(&mut self.rng.0)
    }

    pub fn random_on_range_0_4(&mut self) -> usize {
        self.rng.3.sample(&mut self.rng.0)
    }

    pub fn random_float_0_1(&mut self) -> f64 {
        self.rng.4.sample(&mut self.rng.0)
    }
}

// Private impls
impl Board {
    fn count_words(&mut self) -> (f64, usize) {
        let mut sum = 0.0;
        // Initialize all tiles to NotUsed and when it is iterated through, mark it as word or crap
        for (p, id) in self.priorities.iter_mut().zip(self.placed.iter()) {
            if id.0 >= 2 {
                *p = TilePriority::NotUsed; // Temporary value
            }
        }

        let rows = self.table.axis_iter(Axis(0)).map(|r| r.into_iter().skip(0)).skip(2);
        let cols = self.table.axis_iter(Axis(1)).map(|c| c.into_iter().skip(2)).skip(0);
        for line in rows.chain(cols) {
            // Count words and word stumps. Also keep track what indices those letters are
            let mut words = Vec::<Vec<char>>::new();
            let mut idss = Vec::<Vec<usize>>::new();
            let mut prev = false; // test if last iteration was part of word
            for tile in line {
                match tile {
                    Tile::Occupied { idx, letter } if !prev => {
                        prev = true;
                        let mut word = Vec::with_capacity(MAX_WORD_LENGTH);
                        let mut ids = Vec::with_capacity(MAX_WORD_LENGTH);
                        word.push(*letter);
                        ids.push(*idx);
                        words.push(word);
                        idss.push(ids);
                    }
                    Tile::Occupied { idx, letter } if prev => {
                        words.last_mut().unwrap().push(*letter);
                        idss.last_mut().unwrap().push(*idx);
                    }
                    Tile::Empty if prev => {
                        prev = false;
                    }
                    _ => (),
                }
            }

            for (word, ids) in words.into_iter().zip(idss.into_iter()) {
                let word: String = word.into_iter().collect();

                if word.len() == 1 {
                    continue;
                }

                let is_whole_word = self.dictionaries[0].contains(&word);
                let is_partial_word = self.dictionaries[1].contains(&word);

                let c = self.constants;

                if is_whole_word {
                    sum += ((word.len()-1) as f64).powi(2) * (c[0] as f64 / 10.0);
                    for id in ids {
                        let pr = &mut self.priorities[id];
                        if *pr == TilePriority::NotUsed {
                            *pr = TilePriority::Word;
                        }
                    }
                    continue;
                } else if is_partial_word {
                    sum += ((word.len()-1) as f64).powi(3).sqrt() * (c[1] as f64 / 10.0);
                    for id in ids {
                        let pr = &mut self.priorities[id];
                        if *pr == TilePriority::NotUsed {
                            *pr = TilePriority::PartialWord;
                        }
                    }
                } else {
                    sum -= (word.len()-1) as f64;
                    for id in ids {
                        let pr = &mut self.priorities[id];
                        *pr = TilePriority::Crap;
                    }
                }
            }
        }
        let mut valid_letters = 0;
        for (id, p) in self.priorities.iter_mut().enumerate() {
            match p {
                TilePriority::Word => { valid_letters += 1; },
                TilePriority::NotUsed => {
                    if self.placed[id].0 >= 2 {
                        *p = TilePriority::Crap;
                    }
                },
                _ => (),
            }
        }
        return (sum, valid_letters)
    }

    fn calc_fitness(&mut self) {
        // Count letters, and evaluate fitness for that.
        let (s1, valid_letters) = self.count_words();

        // Punish for placing letters in one big clump.
        let s2 = {
            let mut sum = 0.0;
            for win in self.table.windows((2, 2)) {
                if win.iter().all(|tile| tile.is_occupied()) {
                    sum -= 8.0;
                }
            }
            for win in self.table.windows((3, 3)) {
                let tiles = win.iter().filter(|tile| tile.is_occupied()).count();
                if tiles > 5 {
                    sum -= tiles as f64;
                }
            }
            sum
        };

        // Check if letters are connected on board
        let (parts, s3) = {
            let mut edges = Vec::with_capacity(3 * self.letters.len());
            for win in self.table.windows((1, 2)) {
                if let (
                    Tile::Occupied { idx: a, letter: _ },
                    Tile::Occupied { idx: b, letter: _ }
                ) = (win[(0, 0)], win[(0, 1)])
                {
                    if self.placed[a].0 >= 2 {
                        edges.push((a as u32, b as u32))
                    }
                }
            }
            for win in self.table.windows((2, 1)) {
                if let (
                    Tile::Occupied { idx: a, letter: _ },
                    Tile::Occupied { idx: c, letter: _ }
                ) = (win[(0, 0)], win[(1, 0)])
                {
                    if self.placed[a].0 >= 2 {
                        edges.push((a as u32, c as u32))
                    }
                }
            }
            let graph = UnGraph::<u32, ()>::from_edges(edges);
            let parts = tarjan_scc(&graph);
            let square_sum: usize = parts.iter().map(|v| v.len().pow(2)).sum();
            let sum = self.letters.len().pow(2) - square_sum;
            (parts, -(sum as f64) / 10.0)
        };

        let fitness = self.letters.len().pow(3) as f64 - s1 - s2 - s3;

        let solution = (valid_letters == self.letters.len()) && (parts.len() == 1);
        self.fitness = (fitness, valid_letters, solution);
    }

    fn centerize(&mut self) -> bool {
        let mut min_y = isize::MAX;
        let mut min_x = isize::MAX;
        let mut max_y = isize::MIN;
        let mut max_x = isize::MIN;
        for pos in self.placed.iter().filter(|(y, _)| *y >= 2) {
            min_y = min_y.min(pos.0 as isize);
            min_x = min_x.min(pos.1 as isize);
            max_y = max_y.max(pos.0 as isize);
            max_x = max_x.max(pos.1 as isize);
        }
        let side = self.table.shape()[0] as isize;
        let center_y = side / 2 + 1;
        let center_x = side / 2;
        let center_y_old =  (max_y + min_y) / 2;
        let center_x_old =  (max_x + min_x) / 2;
        let diff_y = center_y_old - center_y;
        let diff_x = center_x_old - center_x;
        if (diff_y * diff_y > 1) && (diff_x * diff_x > 1) {
            self.table.indexed_iter_mut()
                .filter(|((y, _), _)| *y >= 2)
                .for_each(|((_, _), t)| *t = Tile::Empty);
            for (idx, pos) in self.placed.iter_mut()
                .enumerate()
                .filter(|(_, (y, _))| *y >= 2)
            {
                let (y, x) = (pos.0 as isize - diff_y, pos.1 as isize - diff_x);
                assert!(
                    ((2 <= y) && (y < side)) && ((0 <= x) && (x < side)),
                    "Out of bounds idx. {}, ({} {}), ({} {}), ({} {})\n\
                    ({}, {} {}), ({}, {} {}), ({:?})",
                    side, y, x, center_y_old, center_x_old, center_y, center_x,
                    diff_y, min_y, max_y, diff_x, min_x, max_x, pos
                );
                let (y, x): (usize, usize) = (y.try_into().unwrap(), x.try_into().unwrap());
                self.table[(y,x)] = Tile::Occupied {idx, letter: self.letters[idx]};
                //self.table[*pos] = Tile::Empty;
                //println!("{} {}, {:?}, {:?}", idx, self.letters[idx], (y,x), pos);
                *pos = (y,x);
            }
            return true;
        } else {
            return false;
        }
    }

    /// Check that cross referencing indices are valid between `table` and `placed`
    fn invariant(&self) -> Res<()> {
        assert!(self.letters.len() == self.placed.len());
        assert!(self.letters.len() == self.priorities.len());
        let mut check_duplicates = HashSet::new();
        for (i, &(y, x)) in self.placed.iter().enumerate() {
            match self.table[[y,x]] {
                Tile::Occupied{ idx, letter } => {
                    if idx != i || self.letters[i] != letter {
                        self.draw();
                        return Err(anyhow!(
                        "Tile ({}, {}) of `table` doesn't match ({}, {}) of `placed` and `letters`",
                        idx, letter, i, self.letters[i]
                        ));
                    }
                    check_duplicates.insert((x,y));
                }
                Tile::Empty => {
                    self.draw();
                    return Err(anyhow!("Pos {}: ({}, {}) of `placed` does not match `table`.", i, y, x));
                }
            }
        }
        if check_duplicates.len() != self.placed.len() {
            self.draw();
            return Err(anyhow!("Some two items of `placed` point to the same tile"));
        }
        for (id, pr) in self.priorities.iter().enumerate() {
            let pos = self.placed[id];
            match pr {
                TilePriority::NotUsed => {
                    if pos.0 >= 2 {
                        return Err(anyhow!("Tile priority `NotUsed` at pos Pos {:?}", pos));
                    }
                },
                _ => {
                    if pos.0 < 2 {
                        return Err(
                            anyhow!("Some other tile priority than `NotUsed` at pos Pos {:?}", pos)
                        );
                    }
                }
            }
        }
        let tile_count = self.table.iter().filter(|tile| tile.is_occupied()).count();
        if tile_count != self.placed.len() {
            self.draw();
            return Err(anyhow!("There are extra tiles on board!"));
        }
        return Ok(());
    }

    fn dicts_for_word_parts() -> Res<[HashSet<String>; 2]> {
        let set1 = Self::read_all_words_from_csv()?;

        let mut set2 = HashSet::<String>::with_capacity(20*set1.len());
        for word_part_length in 2..=MAX_WORD_LENGTH-1 {
            //let parts = Vec::with_capacity(100_000);
            for word in set1.iter() {
                if word.len() <= word_part_length {
                    continue;
                }
                let mut start = word.char_indices().map(|(i,_)| i);
                let mut end = start.clone()
                    .skip(word_part_length)
                    .chain(std::iter::once(word.len()));
                while let (Some(s), Some(e)) = (start.next(), end.next()) {
                    set2.insert(word[s..e].to_string());
                }
            }
        }
        return Ok([set1, set2]);
    }

    fn read_all_words_from_csv() -> Res<HashSet<String>> {
        use std::{fs::File, io::{BufReader, BufRead as _}};
        let file = File::open("finnish_words.csv")?;
        let mut buff_rdr = BufReader::new(file);
        // Drop out comment header from csv
        for _ in 0..5 {  // Skip header
            let mut header_ = String::new();
            buff_rdr.read_line(&mut header_)?;
            assert!(&header_[0..1] == "#")
        }
        let mut rdr = csv::Reader::from_reader(buff_rdr);
        let chars: Vec<_> = "abcdefghijklmnopqrstuvwxyzäöå".chars().collect();

        let all_words = rdr.records()
            .map(|r| r.ok().and_then(|r| r.get(0).map(|s| s.to_string())).ok_or(()))
            .filter( // Drop all words that are longer than 16 characters or contains special chars
                |s| s.as_ref()
                    .map_or(true, |s| {
                        s.chars().all(|c| chars.contains(&c)) && (s.len() <= MAX_WORD_LENGTH)
                    })
            )
            .collect::<Result<HashSet<String>,()>>()
            .map_err(|_| anyhow!("Error occured in reading csv file."))?;
        return Ok(all_words);
    }
}

/// Loads pre-generated result from disk, or otherwise generates it from closure
fn load_or_generate<T, P, F>(path: P, f: F) -> Res<T>
    where
        P: AsRef<Path>,
        F: FnOnce() -> Res<T>,
        T: Serialize + DeserializeOwned,
{
    use std::{fs::File, io::{Read, Write, BufReader, BufWriter}};
    let path = path.as_ref();
    // Deserialize
    if path.is_file() {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buffer = Vec::<u8>::with_capacity(1_000_000);
        reader.read_to_end(&mut buffer)?;
        return Ok(bincode::deserialize(&buffer[..])?);
    }
    // Generate data
    let dictionaries = f()?;
    // Serialize and write on disk
    let encoded: Vec<u8> = bincode::serialize(&dictionaries).unwrap();
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(encoded.as_slice())?;
    return Ok(dictionaries);
}


// If you want to use Bloom filter, then add following lines to `dicts_for_word_parts()`. Bloom
// filter is way more space efficient than hashset, but I tested it below to be about 30% slower.
//
// use probabilistic_collections::bloom::BloomFilter;
// let mut filter1 = BloomFilter::<String>::new(all_words.len(), 0.001);
// for name in set1.iter() {
//     filter1.insert(name);
// };
// let mut filter2 = BloomFilter::<String>::new(set.len(), 0.001);
// for name in set2.iter() {
//     filter2.insert(name);
// };

// pub fn stress_test() {
//     let all_words = read_csv().unwrap();
//     let (set, [_,dict]) = build_dicts_for_word_parts(all_words);
//     let chars: Vec<_> = "abcdefghijklmnopqrstuvxyzäö".chars().collect();
//     let before = std::time::Instant::now();
//     let mut sum = 0;
//     for i in 0..chars.len() {
//         for j in 0..chars.len() {
//             for k in 0..chars.len() {
//                 for w in 0..chars.len() {
//                     let word: String = [chars[i], chars[j], chars[k], chars[w]].iter().collect();
//                     if dict.contains(&word) {
//                         sum += 1;
//                     }
//                 }
//             }
//         }
//     }
//     println!("With bloom: {} ms, sum: {}", before.elapsed().as_millis(), sum);
//     let mut sum = 0;
//     let before = std::time::Instant::now();
//     for i in 0..chars.len() {
//         for j in 0..chars.len() {
//             for k in 0..chars.len() {
//                 for w in 0..chars.len() {
//                     let word: String = [chars[i], chars[j], chars[k], chars[w]].iter().collect();
//                     if set.contains(&word) {
//                         sum += 1;
//                     }
//                 }
//             }
//         }
//     }
//     println!("With hasset: {} ms, sum: {}", before.elapsed().as_millis(), sum);
//
// }
