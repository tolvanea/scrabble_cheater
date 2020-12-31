use crate::my_prelude::*;
use std::collections::HashSet;
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;
pub use rand::{thread_rng, seq::SliceRandom, Rng, rngs::StdRng, SeedableRng};

const MAX_WORD_LENGTH: usize = 16;

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
        match self {
            Tile::Empty => false,
            _ => true
        }
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
    /// Random number generator
    pub rng: StdRng,
    /// With dictionaries one can test whether string is a real word or part of a real word.
    /// Element [0] is dictionary for full words, element [1] is one for partial words with
    /// lengths 2-16 parts with length 2 and so on.
    pub dictionaries: [HashSet<String>; 2],
}

// Public impls
impl Board {
    pub fn new(side_length: usize, letters: Vec<char>) -> Board {
        assert!(
            (letters.len() < side_length * 2) && (side_length >= 5) && (letters.len() >= 2),
            "Sorry, `side_length` must be at least half of the letter count. \n\
            It is for board reservation of unused letter.",
        );
        let random_seed = thread_rng().gen::<u64>(); // random seed is used by default
        let rng: StdRng = SeedableRng::seed_from_u64(random_seed);

        let path = "dictionaries.serde";
        let dictionaries = load_or_generate(path, dicts_for_word_parts).unwrap();
        let table = Array2::from_elem((side_length, side_length), Tile::Empty);
        // self.letters corresponds indices with self.placed
        let placed = Vec::new();
        let priorities = Vec::new();

        let mut board = Board{
            table,
            letters,
            placed,
            priorities,
            rng,
            dictionaries,
        };
        board.initialize();
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
        let id = self.rng.gen_range(0..self.letters.len());
        let new_pos = (self.table.shape()[0] / 2 + 1, self.table.shape()[0] / 2 + 1);
        let old_pos = &mut self.placed[id];
        self.table[new_pos] = self.table[*old_pos];
        self.table[*old_pos] = Tile::Empty;
        *old_pos = new_pos;
        self.priorities[id] = TilePriority::Crap;
    }

    pub fn fitness(&mut self) -> (f64, usize) {
        let mut sum = 0.0;
        let mut contains_full_words = false;
        let mut contains_crap = false;

        // let mut rows = self.table.outer_iter().enumerate()
        //     .flat_map(|(i, r)| r.iter().enumerate().map(|(j, t)| ((i, j), t)));
        // let mut cols = self.table.axis_iter(Axis(1)).enumerate()
        //     .flat_map(|(j, c)| c.iter().enumerate().map(|(i, t)| ((i, j), t)));
        let rows = self.table.outer_iter().map(|r| r.into_iter().skip(0)).skip(2);
        let cols = self.table.axis_iter(Axis(1)).map(|c| c.into_iter().skip(2)).skip(0);
        for line in rows.chain(cols) {
            // Count words and word stumps. Also keep track what indices those letters are
            let mut words = Vec::<Vec<char>>::with_capacity(MAX_WORD_LENGTH);
            let mut ids = Vec::<Vec<usize>>::with_capacity(MAX_WORD_LENGTH);
            let mut prev = false; // test if last iteration was part of word
            for tile in line {
                match tile {
                    Tile::Occupied { idx, letter } if !prev => {
                        prev = true;
                        words.push(vec![*letter]);
                        ids.push(vec![*idx]);
                        self.priorities[*idx] = TilePriority::Crap;
                    }
                    Tile::Occupied { idx, letter } if prev => {
                        words.last_mut().unwrap().push(*letter);
                        ids.last_mut().unwrap().push(*idx);
                        self.priorities[*idx] = TilePriority::Crap;
                    }
                    Tile::Empty if prev => {
                        prev = false;
                    }
                    _ => {},
                }
            }

            for (word, ids) in words.into_iter().zip(ids.into_iter()) {
                let word: String = word.into_iter().collect();
                let is_whole_word = self.dictionaries[0].contains(&word);
                let is_partial_word = self.dictionaries[1].contains(&word);

                if is_whole_word {
                    sum += ((word.len()-1) as f64).powi(2);
                    contains_full_words = true;
                    for id in ids {
                        self.priorities[id] = TilePriority::Word;
                    }
                    continue;
                } else if is_partial_word {
                    sum += ((word.len()-1) as f64).powi(3).sqrt();
                    for id in ids {
                        let pr = &mut self.priorities[id];
                        if *pr != TilePriority::Word {
                            *pr = TilePriority::PartialWord;
                        }
                    }
                } else {
                    sum -= (word.len()-1) as f64;
                    for id in ids {
                        let pr = &mut self.priorities[id];
                        if (*pr != TilePriority::Word) && (*pr != TilePriority::PartialWord) {
                            *pr = TilePriority::Crap;
                        }
                    }
                }
                contains_crap = true;
            }
        }
        // Punish for placing letters in one big clump.
        for win in self.table.windows((2,2)) {
            if win.iter().all(|tile| tile.is_occupied()) {
                sum -= 8.0;
            }
        }
        for win in self.table.windows((3,3)) {
            let tiles = win.iter().filter(|tile| tile.is_occupied()).count();
            if tiles > 5 {
                sum -= tiles as f64;
            }
        }
        let mut letters_in_words = 0;
        for p in self.priorities.iter() {
            if *p == TilePriority::Word {
                letters_in_words += 1;
            }
        }
        if contains_full_words && !contains_crap {
            return (-sum, letters_in_words)  // Solution found!
        } else {
            return (self.letters.len().pow(3) as f64 - sum, letters_in_words);
        }
    }

    pub fn draw(&self) {
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

    /// Check that cross referencing indices are valid between `table` and `placed`
    pub fn invariant(&self) -> Res<()> {
        for (i, &(x, y)) in self.placed.iter().enumerate() {
            match self.table[[x,y]] {
                Tile::Occupied{ idx, letter } => {
                    if idx != i || self.letters[i] != letter {
                        self.draw();
                        return Err(anyhow!(
                        "Tile ({}, {}) of `table` doesn't match ({}, {}) of `placed` and `letters`",
                        idx, letter, i, self.letters[i]
                        ));
                    }
                }
                Tile::Empty => {
                    self.draw();
                    return Err(anyhow!("Pos ({}, {}) of `placed` does not match `table`."));
                }
            }
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
                        return Err(anyhow!("Tile priority is not `NotUsed` at pos Pos {:?}", pos));
                    }
                }
            }
        }
        return Ok(());
    }

    pub fn random_move(&mut self) {
        if self.rng.gen_range(0..1000) == 0 {
            self.invariant();
        }

        let id = loop {
            let id = self.rng.gen_range(0..self.placed.len());
            match self.priorities[id] {
                TilePriority::Crap | TilePriority::NotUsed => break id,
                TilePriority::Word => if self.rng.gen_range(0..20) == 0 { break id; },
                TilePriority::PartialWord => if self.rng.gen_range(0..2) == 0 { break id },
            }
        };


        let old_pos: (usize, usize) = self.placed[id];
        let tile = self.table[old_pos];
        debug_assert!(tile != Tile::Empty);
        self.table[old_pos] = Tile::Empty;
        // Loop until new position has been found
        'outer: loop {
            let side = self.table.shape()[0] as isize;
            let (mut ni, d) = if self.rng.gen_range(0..9) > 0 {
                // Trial position of new tile that is at position of existing tile
                let nu = self.placed[self.rng.gen_range(0..self.letters.len())];
                let ni = (nu.0 as isize, nu.1 as isize);
                // Direction in which new tiles will be searched
                let d = [(-1, 0), (1, 0), (0, 1), (0, -1)][self.rng.gen_range(0..4)];
                (ni, d)
            } else {
                let nu = (self.rng.gen_range(0..2), 0);
                let ni = (nu.0 as isize, nu.1 as isize);
                let d = (0, 1);
                (ni, d)
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
                        return;
                    }
                } else {
                    // Random choice was bad. Choose a new one.
                    continue 'outer
                }
            }
        }
    }
}

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


fn dicts_for_word_parts() -> Res<[HashSet<String>; 2]> {
    let set1 = read_all_words_from_csv()?;

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
            loop {
                if let (Some(s), Some(e)) = (start.next(), end.next()) {
                    set2.insert(word[s..e].to_string());
                }
                else {
                    break
                }
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

    let all_words = rdr.records()
        .map(|r| r.ok().and_then(|r| r.get(0).map(|s| s.to_string())).ok_or(()))
        .filter( // Drop all words that are longer than 16 characters or contains '-'
            |s| s.as_ref().map_or(true, |s| !s.contains('-') && (s.len() <= MAX_WORD_LENGTH))
        )
        .collect::<Result<HashSet<String>,()>>()
        .map_err(|_| anyhow!("Error occured in reading csv file."))?;
    return Ok(all_words);
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
//     let chars: Vec<_> = "abcdefghijklmnopqrstuvwxyzäö".chars().collect();
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
