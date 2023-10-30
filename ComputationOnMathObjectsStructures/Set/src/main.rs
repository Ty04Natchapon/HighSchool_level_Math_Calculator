use std::collections::HashSet;
use std::io::{self, Write}; // Import Write to flush stdout

fn parse_set(set_str: &str) -> Result<HashSet<i32>, std::num::ParseIntError> {
    set_str.split(',')
        .filter(|s| !s.trim().is_empty()) // Filter out empty strings
        .map(str::parse)
        .collect()
}

fn display_format() {
    println!("Please enter the numbers one set at a time:");
    println!("Example:");
    println!(" Set1: 1,2,3,4");
    println!(" Set2: 5,6,7");
}

fn read_set(set_name: &str) -> HashSet<i32> {
    loop {
        print!("Enter {} (e.g., 1,2,3): ", set_name);
        io::stdout().flush().unwrap(); // Flush stdout to ensure prompt appears

        let mut set_input = String::new();
        io::stdin().read_line(&mut set_input).expect("Failed to read line");

        match parse_set(set_input.trim()) {
            Ok(set) => return set,
            Err(_) => {
                eprintln!("Please enter integers separated by commas. ");
                eprintln!("Enter your Set in this format: 1,2,3 ");
            }
        }
    }
}

fn main() {
    display_format();

    let set1 = read_set("Set1");
    let set2 = read_set("Set2");

    println!("Set 1: {:?}", set1);
    println!("Set 2: {:?}", set2);

    // Union
    let union: HashSet<_> = set1.union(&set2).cloned().collect();
    println!("Union: {:?}", union);

    // Intersection
    let intersection: HashSet<_> = set1.intersection(&set2).cloned().collect();
    println!("Intersection: {:?}", intersection);

    // Difference
    let difference: HashSet<_> = set1.difference(&set2).cloned().collect();
    println!("Difference (Set 1 - Set 2): {:?}", difference);

    // Symmetric Difference
    let sym_difference: HashSet<_> = set1.symmetric_difference(&set2).cloned().collect();
    println!("Symmetric Difference: {:?}", sym_difference);

    // Subset and Superset
    if set1.is_subset(&set2) {
        println!("Set 1 is a subset of Set 2");
    } else if set2.is_subset(&set1) {
        println!("Set 2 is a subset of Set 1");
    } else {
        println!("Neither set is a subset of the other");
    }

    if set1.is_superset(&set2) {
        println!("Set 1 is a superset of Set 2");
    } else if set2.is_superset(&set1) {
        println!("Set 2 is a superset of Set 1");
    } else {
        println!("Neither set is a superset of the other");
    }

    // Check if set is empty
    println!("Is Set 1 empty?: {}", set1.is_empty());
    println!("Is Set 2 empty?: {}", set2.is_empty());

    // Find minimum and maximum 
    if !set1.is_empty() {
        println!("Set 1 minimum: {:?}", set1.iter().min().unwrap());
        println!("Set 1 maximum: {:?}", set1.iter().max().unwrap());
    } else {
        println!("Set 1 is empty, so no min/max values");
    }
    
    if !set2.is_empty() {
        println!("Set 2 minimum: {:?}", set2.iter().min().unwrap());
        println!("Set 2 maximum: {:?}", set2.iter().max().unwrap());
    } else {
        println!("Set 2 is empty, so no min/max values");
    }
}
