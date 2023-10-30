//This code might look werid due to i create all of them seperately then combine them together 
use std::collections::HashSet;
use std::io::{self, Write};
use std::fmt;
use std::ops::{Add, Sub, Neg};

fn display_main_menu() {
    println!("======================================");
    println!("=            MAIN MENU               =");
    println!("======================================");
    println!("Enter the corresponding number (1-6) for the operation you'd like to perform");
    println!("1. Vector Operations");
    println!("2. Matrix Operations");
    println!("3. Set Operations"); 
    println!("4. Boolean Logic Evaluator"); 
    println!("5. Complex number(a+i)"); 
    println!("6. Quaternion(w+xi+yj+zk)"); 
    println!("7. Exit"); 
    println!("======================================");
}

fn main() {
    loop {
        display_main_menu();
        let choice = get_integer_input("Enter your choice: ");

        match choice {
            1 => vector(),
            2 => matrix_menu(),
            3 => set_operations(), 
            4 => boolean_menu(),
            5 => complex(),
            6 => quaternion(),
            7 => {
                println!("Thanks for using the calculator. Goodbye!");
                break;
            }
            _ => println!("Invalid choice. Please select a valid option(Number:1-5)."),
        }
    }
}
fn get_integer_input(prompt: &str) -> i32 {
    loop {
        println!("{}", prompt);
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read line");
        
        match input.trim().parse::<i32>() {
            Ok(value) => return value,
            Err(_) => println!("Invalid choice. Please enter correct input."),
        }
    }
}


// ===================
// = Set Operations =
// ===================

fn parse_set(set_str: &str) -> Result<HashSet<i32>, std::num::ParseIntError> {
    set_str.split(',')
        .filter(|s| !s.trim().is_empty())
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
        io::stdout().flush().unwrap();

        let mut set_input = String::new();
        io::stdin().read_line(&mut set_input).expect("Failed to read line");

        match parse_set(set_input.trim()) {
            Ok(set) => return set,
            Err(_) => {
                eprintln!("Please enter integers separated by commas.");
                eprintln!("Enter your Set in this format: 1,2,3");
            }
        }
    }
}

fn set_operations() {
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

// ==========================
// = Boolean Logic Evaluator=
// ==========================
use std::collections::VecDeque;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    Lowest,
    Implication, // -> and <->
    Or,          // ||
    And,         // &&
    Not,         // !
    Highest,
}

#[derive(PartialEq)]
enum Token {
    And,
    Or,
    Not,
    Implies,
    Iff, // if and only if
    True,
    False,
    LeftParen,
    RightParen,
}

fn tokenize(input: &str) -> Vec<Token> {
    input.split_whitespace().filter_map(|s| match s {
        "&&" => Some(Token::And),
        "||" => Some(Token::Or),
        "!" => Some(Token::Not),
        "->" => Some(Token::Implies),
        "<->" => Some(Token::Iff),
        "T" => Some(Token::True),
        "F" => Some(Token::False),
        "(" => Some(Token::LeftParen),
        ")" => Some(Token::RightParen),
        _ => None, // Ignore unknown tokens
    }).collect()
}

fn precedence(token: &Token) -> Precedence {
    match *token {
        Token::Implies | Token::Iff => Precedence::Implication,
        Token::Or => Precedence::Or,
        Token::And => Precedence::And,
        Token::Not => Precedence::Not,
        Token::LeftParen | Token::RightParen => Precedence::Highest,
        _ => Precedence::Lowest,
    }
}

fn shunting_yard(tokens: Vec<Token>) -> Vec<Token> {
    let mut queue = VecDeque::new();
    let mut stack = Vec::new();

    for token in tokens {
        match token {
            Token::True | Token::False => queue.push_back(token),
            Token::LeftParen => stack.push(token),
            Token::RightParen => {
                while let Some(op) = stack.pop() {
                    if op == Token::LeftParen {
                        break;
                    }
                    queue.push_back(op);
                }
            }
            _ => {
                while stack.last().map_or(false, |op| precedence(op) > precedence(&token)) {
                    queue.push_back(stack.pop().unwrap());
                }
                stack.push(token);
            }
        }
    }

    while let Some(op) = stack.pop() {
        queue.push_back(op);
    }

    Vec::from(queue)
}

fn evaluate_rpn(tokens: Vec<Token>) -> Result<bool, &'static str> {
    let mut stack = Vec::new();

    for token in tokens {
        match token {
            Token::True => stack.push(true),
            Token::False => stack.push(false),
            Token::Not => {
                if let Some(operand) = stack.pop() {
                    stack.push(!operand);
                } else {
                    return Err("Invalid syntax: missing operand for !");
                }
            }
            _ => {
                if let (Some(right), Some(left)) = (stack.pop(), stack.pop()) {
                    stack.push(match token {
                        Token::And => left && right,
                        Token::Or => left || right,
                        Token::Implies => !left || right,
                        Token::Iff => left == right,
                        _ => return Err("Invalid syntax: unknown operator"),
                    });
                } else {
                    return Err("Invalid syntax: missing operands");
                }
            }
        }
    }

    if stack.len() == 1 {
        Ok(stack[0])
    } else {
        Err("Please enter correct input")
    }
}

fn evaluate_logic_expression(expression: &str) -> Result<bool, &'static str> {
    let tokens = tokenize(expression);
    let rpn = shunting_yard(tokens);
    evaluate_rpn(rpn)
}

fn boolean_menu() {
    println!("Welcome to the Logic Expression Evaluator!");
    println!("T = True, F = False : Must be uppercase");
    println!("&& = and, || = or, -> = if then <-> = if and only if, ! = not");
    println!("Example use: T -> F || T");

    loop {
        let mut expression = String::new();
        println!("Enter 'exit' to go back to main menu.");
        println!("Enter a logic expression: ");
        
        io::stdin().read_line(&mut expression).expect("Failed to read line");

        let expression = expression.trim();
        if expression == "exit" {
            break;
        }

        match evaluate_logic_expression(expression) {
            Ok(result) => println!("Result: {}", result),
            Err(e) => println!("Error: {}", e),
        }
    }

    println!("Thank you for using the Logic Expression Evaluator!");
}

// ===================
// = Vector Operations =
// ===================

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

fn vector() {
    let mut choice = String::new();
    loop {
        println!("Choose vector dimension: Press 1 for 2D(x,y) or 2 for 3D(x,y,z):");
        io::stdin().read_line(&mut choice).expect("Failed to read line");

        match choice.trim() {
            "1" => {
                if let Some((vector1, vector2)) = get_2d_vectors() {
                    Vector_menu(vector1, vector2);
                    break;
                }
            },
            "2" => {
                if let Some((vector1, vector2)) = get_3d_vectors() {
                    Vector_menu(vector1, vector2);
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
    let values: Vec<f64> = input.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    if values.len() == 4 {
        Some((Vector::new(values[0], values[1], None), Vector::new(values[2], values[3], None)))
    } else {
        println!("Invalid input. Please follow the format.");
        None
    }
}

fn get_3d_vectors() -> Option<(Vector, Vector)> {
    let mut input = String::new();
    println!("Enter 3D vectors as <x1> <y1> <z1> <x2> <y2> <z2>:");
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let values: Vec<f64> = input.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    if values.len() == 6 {
        Some((Vector::new(values[0], values[1], Some(values[2])), Vector::new(values[3], values[4], Some(values[5]))))
    } else {
        println!("Invalid input. Please follow the format.");
        println!("For 2D: <x1> <y1> <x2> <y2>");
        println!("For 3D: <x1> <y1> <z1> <x2> <y2> <z2>");
        None
    }
}

fn Vector_menu(v1: Vector, v2: Vector) {
    loop {
        println!("Vector Operations:");
        println!("1. Addition");
        println!("2. Subtraction");
        println!("3. Dot Product");
        println!("4. Cross Product");
        println!("5. Magnitude of First Vector");
        println!("6. Normalize First Vector");
        println!("7. Distance between two vectors");
        println!("8. Return to Main Menu");

        let choice = get_integer_input("Enter your choice: ");

        match choice {
            1 => println!("Result: {}", v1 + v2),
            2 => println!("Result: {}", v1 - v2),
            3 => println!("Dot Product: {}", v1.dot(&v2)),
            4 => {
                if v1.z.is_some() && v2.z.is_some() {
                    println!("Cross Product: {}", v1.cross(&v2));
                } else {
                    println!("Cross product is only defined for 3D vectors.");
                }
            }
            5 => println!("Magnitude of Vector1: {}", v1.magnitude()),
            6 => match v1.normalize() {
                Some(v) => println!("Normalized Vector1: {}", v),
                None => println!("Cannot normalize a zero vector."),
            },
            7 => println!("Distance between vectors: {}", v1.distance_to(&v2)),
            8 => break,
            _ => println!("Invalid choice. Please select a valid option(1 or 2)."),
        }
    }
}

// ===================
// = Matrix Operations =
// ===================

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

fn get_integer_input_matrix(prompt: &str) -> usize {
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
    let rows= get_integer_input_matrix("Enter number of rows: ");
    let cols = get_integer_input_matrix("Enter number of columns: ");
    (rows, cols)
}

fn matrix_menu() {
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
        println!("6. Back to main menu");

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
// ===================
// = Complex Operations =
// ===================

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

fn complex() {
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
                println!("For example if you want to type 1 + 2i, enter: 1,2");
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


// ===================
// = Quertinion Operations =
// ===================

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

fn quaternion() {
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