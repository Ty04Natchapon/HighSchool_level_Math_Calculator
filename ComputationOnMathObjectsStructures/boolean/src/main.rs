    use std::io;
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
            Err("Invalid syntax: the stack should have exactly one boolean left")
        }
    }

    fn evaluate_logic_expression(expression: &str) -> Result<bool, &'static str> {
        let tokens = tokenize(expression);
        let rpn = shunting_yard(tokens);
        evaluate_rpn(rpn)
    }

    fn main() {
        println!("Welcome to the Logic Expression Evaluator!");
        println!("&& = and, || = or, -> = if then <-> = if and only if, ! = not");
        println!("Enter 'exit' to quit.");

        loop {
            let mut expression = String::new();
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

    