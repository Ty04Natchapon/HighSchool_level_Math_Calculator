use std::fmt;
use std::ops::{Add, Sub, Neg};
use std::io;

#[derive(Debug, Clone, Copy)]
struct Vector {
    x: f64,
    y: f64,
    z: Option<f64>,
}

impl Vector {
    fn new(x: f64, y: f64, z: Option<f64>) -> Vector {
        Vector { x, y, z }
    }

    fn dot(&self, other: &Vector) -> f64 {
        let z_dot = self.z.unwrap_or(0.0) * other.z.unwrap_or(0.0);
        self.x * other.x + self.y * other.y + z_dot
    }

    fn cross(&self, other: &Vector) -> Vector {
        let zx = self.z.unwrap_or(0.0);
        let zy = other.z.unwrap_or(0.0);
        
        Vector::new(
            self.y * zy - zx * other.y,
            zx * other.x - self.x * zy,
            Some(self.x * other.y - self.y * other.x),
        )
    }

    fn magnitude(&self) -> f64 {
        let z2 = self.z.unwrap_or(0.0).powi(2);
        (self.x * self.x + self.y * self.y + z2).sqrt()
    }

    fn normalize(&self) -> Option<Vector> {
        let mag = self.magnitude();
        if mag > 0.0 {
            Some(Vector::new(self.x / mag, self.y / mag, self.z.map(|z| z / mag)))
        } else {
            None
        }
    }
    

    fn distance_to(&self, other: &Vector) -> f64 {
        let z_diff = self.z.unwrap_or(0.0) - other.z.unwrap_or(0.0);
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + z_diff.powi(2)).sqrt()
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.z {
            Some(z) => write!(f, "({}, {}, {})", self.x, self.y, z),
            None => write!(f, "({}, {})", self.x, self.y),
        }
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vector::new(self.x + other.x, self.y + other.y, self.z.zip(other.z).map(|(a, b)| a + b))
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vector::new(self.x - other.x, self.y - other.y, self.z.zip(other.z).map(|(a, b)| a - b))
    }
}

impl Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector::new(-self.x, -self.y, self.z.map(|z| -z))
    }
}

fn main() {
    let mut choice = String::new();
    loop {
        println!("Choose vector dimension: Press 1 for 2D(x,y) or 2 for 3D(x,y,z):");
        io::stdin().read_line(&mut choice).expect("Failed to read line");

        match choice.trim() {
            "1" => {
                if let Some((vector1, vector2)) = get_2d_vectors() {
                    perform_operations(vector1, vector2);
                    break;
                }
            },
            "2" => {
                if let Some((vector1, vector2)) = get_3d_vectors() {
                    perform_operations(vector1, vector2);
                    break;
                }
            },
            _ => {
                println!("Invalid choice! Press 1 for 2D or 2 for 3D.");
            }
        }
        choice.clear();
    }
}

fn get_2d_vectors() -> Option<(Vector, Vector)> {
    let mut input = String::new();
    println!("Enter 2D vectors as <x1> <y1> <x2> <y2>:");
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let values: Vec<f64> = input
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap_or_default())
        .collect();

    if values.len() != 4 {
        println!("Expected 4 values for 2D vectors. Please try again.");
        return None;
    }

    Some((
        Vector::new(values[0], values[1], None),
        Vector::new(values[2], values[3], None)
    ))
}

fn get_3d_vectors() -> Option<(Vector, Vector)> {
    let mut input = String::new();
    println!("Enter 3D vectors as <x1> <y1> <z1> <x2> <y2> <z2>:");
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let values: Vec<f64> = input
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap_or_default())
        .collect();

    if values.len() != 6 {
        println!("Expected 6 values for 3D vectors. Please try again.");
        return None;
    }

    Some((
        Vector::new(values[0], values[1], Some(values[2])),
        Vector::new(values[3], values[4], Some(values[5]))
    ))
}

fn perform_operations(v1: Vector, v2: Vector) {
    println!("Vector 1: {}", v1);
    println!("Vector 2: {}", v2);

    let sum = v1 + v2;
    let difference = v1 - v2;
    let dot_product = v1.dot(&v2);
    let cross_product = v1.cross(&v2);
    let normalized = v1.normalize().expect("Cannot normalize zero vector");
    let distance = v1.distance_to(&v2);

    println!("Sum: {}", sum);
    println!("Difference: {}", difference);
    println!("Dot Product: {}", dot_product);
    println!("Cross Product: {}", cross_product);
    println!("Normalized Vector 1: {}", normalized);
    println!("Distance between points: {}", distance);
    println!("Magnitude of Vector 1: {}", v1.magnitude());
}
