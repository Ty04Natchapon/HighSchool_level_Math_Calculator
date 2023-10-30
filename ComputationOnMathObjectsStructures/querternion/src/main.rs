use std::io;

#[derive(Debug, Clone, Copy)]
struct Quaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quaternion {
    fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quaternion { w, x, y, z }
    }

    fn add(self, other: Quaternion) -> Quaternion {
        Quaternion::new(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn sub(self, other: Quaternion) -> Quaternion {
        Quaternion::new(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn mul(self, other: Quaternion) -> Quaternion {
        Quaternion::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }

    fn magnitude(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn conjugate(self) -> Quaternion {
        Quaternion::new(self.w, -self.x, -self.y, -self.z)
    }

    fn inverse(self) -> Option<Quaternion> {
        let mag_squared = self.magnitude().powi(2);
        if mag_squared == 0.0 {
            None
        } else {
            Some(Quaternion {
                w: self.w / mag_squared,
                x: -self.x / mag_squared,
                y: -self.y / mag_squared,
                z: -self.z / mag_squared,
            })
        }
    }
    
}

impl std::ops::Mul<Quaternion> for Quaternion {
    type Output = Self;

    fn mul(self, other: Quaternion) -> Self {
        self.mul(other)
    }
}

fn parse_quaternion(input: &str) -> Result<Quaternion, &'static str> {
    let parts: Vec<&str> = input.split(',').collect();
    if parts.len() != 4 {
        return Err("Invalid quaternion format. Expecting w,x,y,z");
    }

    let w = parts[0].parse().map_err(|_| "Failed to parse w component")?;
    let x = parts[1].parse().map_err(|_| "Failed to parse x component")?;
    let y = parts[2].parse().map_err(|_| "Failed to parse y component")?;
    let z = parts[3].parse().map_err(|_| "Failed to parse z component")?;

    Ok(Quaternion::new(w, x, y, z))
}

fn main() {
    loop {
        println!("Choose an operation:");
        println!("1: Addition (+)");
        println!("2: Subtraction (-)");
        println!("3: Multiplication (*)");
        println!("4: Magnitude");
        println!("5: Conjugate");
        println!("6: Inverse");
        println!("0: Exit");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).expect("Failed to read line");
        let choice = choice.trim();

        match choice {
            "1" => {
                println!("Enter the first quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();

                println!("Enter the second quaternion (format: w,x,y,z):");
                let q2 = read_quaternion();

                let result = q1.add(q2);
                println!("Result: {},{},{},{}", result.w, result.x, result.y, result.z);
            }
            "2" => {
                println!("Enter the first quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();

                println!("Enter the second quaternion (format: w,x,y,z):");
                let q2 = read_quaternion();

                let result = q1.sub(q2);
                println!("Result: {},{},{},{}", result.w, result.x, result.y, result.z);
            }
            "3" => {
                println!("Enter the first quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();

                println!("Enter the second quaternion (format: w,x,y,z):");
                let q2 = read_quaternion();

                let result = q1 * q2;
                println!("Result: {},{},{},{}", result.w, result.x, result.y, result.z);
            }
            "4" => {
                println!("Enter the quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();
                println!("Result: {}", q1.magnitude());
            }
            "5" => {
                println!("Enter the quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();
                let result = q1.conjugate();
                println!("Result: {},{},{},{}", result.w, result.x, result.y, result.z);
            }
            "6" => {
                println!("Enter the quaternion (format: w,x,y,z):");
                let q1 = read_quaternion();
                match q1.inverse() {
                    Some(result) => println!("Result: {},{},{},{}", result.w, result.x, result.y, result.z),
                    None => println!("Quaternion has no inverse."),
                }
            }
            "0" => break,
            _ => println!("Unsupported choice."),
        }
    }
}



fn read_quaternion() -> Quaternion {
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        match parse_quaternion(&input.trim()) {
            Ok(q) => return q,
            Err(_) => println!("Please enter correct input. Format: w,x,y,z"),
        }
    }
}

fn read_scalar() -> f64 {
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        match input.trim().parse::<f64>() {
            Ok(scalar) => return scalar,
            Err(_) => println!("Please enter correct input for a scalar value."),
        }
    }
}

fn read_vector3() -> Vec<f64> {
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        let parts: Vec<Result<f64, _>> = input.trim().split(',').map(|s| s.parse()).collect();
        if parts.len() == 3 && parts.iter().all(Result::is_ok) {
            return parts.into_iter().map(Result::unwrap).collect();
        } else {
            println!("Please enter correct input. Format: ux,uy,uz");
        }
    }
}