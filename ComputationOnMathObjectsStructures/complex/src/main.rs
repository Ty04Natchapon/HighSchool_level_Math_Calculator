use std::io;

#[derive(Debug, Clone, Copy)]
struct Complex {
    r: f64,
    i: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { r: real, i: imag }
    }

    fn add(self, other: Complex) -> Complex {
        Complex::new(self.r + other.r, self.i + other.i)
    }

    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.r - other.r, self.i - other.i)
    }

    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.r * other.r - self.i * other.i,
            self.r * other.i + self.i * other.r
        )
    }

    fn magnitude(self) -> f64 {
        (self.r * self.r + self.i * self.i).sqrt()
    }

    fn conjugate(self) -> Complex {
        Complex::new(self.r, -self.i)
    }
}

fn parse_complex(input: &str) -> Result<Complex, &'static str> {
    let parts: Vec<&str> = input.split(',').collect();
    if parts.len() != 2 {
        return Err("Invalid complex format. Expecting real,imag");
    }

    let real = parts[0].parse().map_err(|_| "Failed to parse real component")?;
    let imag = parts[1].parse().map_err(|_| "Failed to parse imaginary component")?;

    Ok(Complex::new(real, imag))
}

fn main() {
    loop {
        println!("Choose an operation:");
        println!("1: Addition (+)");
        println!("2: Subtraction (-)");
        println!("3: Multiplication (*)");
        println!("4: Magnitude");
        println!("5: Conjugate");
        println!("0: Exit");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).expect("Failed to read line");
        let choice = choice.trim();

        match choice {
            "1" => {
                println!("Enter the first complex number (format: real,imag):");
                println!("For example if you want to type 1 + 2i, enter: 1,2");
                let c1 = read_complex();

                println!("Enter the second complex number (format: real,imag):");
                println!("For example if you want to type 1 - 2i, enter: 1,2");
                let c2 = read_complex();

                let result = c1.add(c2);
                println!("Result: {} + {}i", result.r, result.i);
            }
            "2" => {
                println!("Enter the first complex number (format: real,imag):");
                let c1 = read_complex();

                println!("Enter the second complex number (format: real,imag):");
                let c2 = read_complex();

                let result = c1.sub(c2);
                println!("Result: {} + {}i", result.r, result.i);
            }
            "3" => {
                println!("Enter the first complex number (format: real,imag):");
                let c1 = read_complex();

                println!("Enter the second complex number (format: real,imag):");
                let c2 = read_complex();

                let result = c1.mul(c2);
                println!("Result: {} + {}i", result.r, result.i);
            }
            "4" => {
                println!("Enter the complex number (format: real,imag):");
                let c1 = read_complex();
                println!("Result: {} + {}i", c1.r, c1.i);
            }
            "5" => {
                println!("Enter the complex number (format: real,imag):");
                let c1 = read_complex();
                let result = c1.conjugate();
                println!("Result: {} + {}i", result.r, result.i);
            }
            "0" => break,
            _ => println!("Unsupported choice."),
        }
    }
}


fn read_complex() -> Complex {
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        match parse_complex(&input.trim()) {
            Ok(c) => return c,
            Err(_) => println!("Please enter correct input. Format: real,imag"),
        }
    }
}
