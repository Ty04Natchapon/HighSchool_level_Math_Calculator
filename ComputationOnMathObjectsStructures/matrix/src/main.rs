use std::io::{self, Write};

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            Err("Data size doesn't match matrix dimensions")
        } else {
            Ok(Matrix { rows, cols, data })
        }
    }

    fn add(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            Err("Matrix dimensions do not match for addition")
        } else {
            let result_data = self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect();
            Ok(Matrix::new(self.rows, self.cols, result_data).unwrap())
        }
    }

    fn subtract(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            Err("Matrix dimensions do not match for subtraction")
        } else {
            let result_data = self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect();
            Ok(Matrix::new(self.rows, self.cols, result_data).unwrap())
        }
    }

    fn multiply(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.cols != other.rows {
            Err("Matrix dimensions are not compatible for multiplication")
        } else {
            let mut result_data = vec![0.0; self.rows * other.cols];
            for i in 0..self.rows {
                for j in 0..other.cols {
                    for k in 0..self.cols {
                        result_data[i * other.cols + j] +=
                            self.data[i * self.cols + k] * other.data[k * other.cols + j];
                    }
                }
            }
            Ok(Matrix::new(self.rows, other.cols, result_data).unwrap())
        }
    }

    fn transpose(&self) -> Self {
        let mut result_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix::new(self.cols, self.rows, result_data).unwrap()
    }

    fn determinant(&self) -> Result<f64, &'static str> {
        if self.rows == 2 && self.cols == 2 {
            let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];
            Ok(det)
        } else if self.rows == 3 && self.cols == 3 {
            let det = self.data[0] * (self.data[4] * self.data[8] - self.data[5] * self.data[7])
                - self.data[1] * (self.data[3] * self.data[8] - self.data[5] * self.data[6])
                + self.data[2] * (self.data[3] * self.data[7] - self.data[4] * self.data[6]);
            Ok(det)
        } else {
            Err("Matrix size not supported for determinant")
        }
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j == self.cols - 1 {
                    write!(f, "{:5.1}", self.data[i * self.cols + j])?;
                } else {
                    write!(f, "{:5.1},", self.data[i * self.cols + j])?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

fn get_input(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn get_integer_input(prompt: &str) -> usize {
    loop {
        let input = get_input(prompt);
        match input.parse::<usize>() {
            Ok(val) => return val,
            Err(_) => println!("Please enter input as an integer."),
        }
    }
}

fn get_float_input(prompt: &str) -> f64 {
    loop {
        let input = get_input(prompt);
        match input.parse::<f64>() {
            Ok(val) => return val,
            Err(_) => println!("Please enter a valid number."),
        }
    }
}

fn get_matrix_input(rows: usize, cols: usize) -> Matrix {
    let mut data = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let value = get_float_input(&format!("Enter element [{}][{}]: ", i + 1, j + 1));
            data.push(value);
        }
    }

    Matrix::new(rows, cols, data).unwrap()
}

fn get_matrix_dimensions() -> (usize, usize) {
    let rows = get_integer_input("Enter number of rows: ");
    let cols = get_integer_input("Enter number of columns: ");
    (rows, cols)
}

fn main() {
    println!("Matrix Calculator");

    println!("Enter dimensions for Matrix 1:");
    let (rows1, cols1) = get_matrix_dimensions();
    println!("Enter Matrix 1:");
    let matrix1 = get_matrix_input(rows1, cols1);
    println!("Matrix 1:\n{}", matrix1);

    println!("Enter dimensions for Matrix 2:");
    let (rows2, cols2) = get_matrix_dimensions();
    println!("Enter Matrix 2:");
    let matrix2 = get_matrix_input(rows2, cols2);
    println!("Matrix 2:\n{}", matrix2);


    loop {
        println!("\nChoose operation:");
        println!("1. Add");
        println!("2. Subtract");
        println!("3. Multiply");
        println!("4. Transpose Matrix 1");
        println!("5. Determinant of Matrix 1 (only for 2x2 and 3x3)");
        println!("6. Exit");

        let operation = get_integer_input("Enter choice (1/2/3/4/5/6): ");

        match operation {
            1 => {
                match matrix1.add(&matrix2) {
                    Ok(result) => println!("Result:\n{}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            2 => {
                match matrix1.subtract(&matrix2) {
                    Ok(result) => println!("Result:\n{}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            3 => {
                match matrix1.multiply(&matrix2) {
                    Ok(result) => println!("Result:\n{}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            4 => {
                let result = matrix1.transpose();
                println!("Result:\n{}", result);
            }
            5 => {
                match matrix1.determinant() {
                    Ok(det) => println!("Determinant: {}", det),
                    Err(e) => println!("Error: {}", e),
                }
            }
            6 => break,
            _ => println!("Invalid choice."),
        }
    }
}