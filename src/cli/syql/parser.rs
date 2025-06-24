//! SyQL Parser Implementation
//!
//! This module implements a sophisticated parser for the SyQL query language using a
//! recursive descent parser with error recovery and auto-completion support.

use super::ast::*;
use super::{CompletionItem, CompletionType};
use crate::error::{Result, MemoryError};
use std::collections::HashMap;



/// SyQL parser
pub struct SyQLParser {
    /// Keywords for completion
    keywords: Vec<&'static str>,
    /// Functions for completion
    functions: Vec<&'static str>,
    /// Operators for completion
    _operators: Vec<&'static str>,
}

impl SyQLParser {
    /// Create a new SyQL parser
    pub fn new() -> Result<Self> {
        Ok(Self {
            keywords: vec![
                "SELECT", "MATCH", "FROM", "WHERE", "ORDER", "BY", "LIMIT", "OFFSET",
                "GROUP", "HAVING", "WITH", "AS", "DISTINCT", "CREATE", "UPDATE", "DELETE",
                "SET", "INSERT", "INTO", "VALUES", "SHOW", "EXPLAIN", "DESCRIBE",
                "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "REGEX", "EXISTS",
                "CASE", "WHEN", "THEN", "ELSE", "END", "IF", "NULL", "TRUE", "FALSE",
                "ASC", "DESC", "NULLS", "FIRST", "LAST", "INNER", "LEFT", "RIGHT",
                "FULL", "CROSS", "JOIN", "ON", "USING", "UNION", "INTERSECT", "EXCEPT",
                "PATH", "SHORTEST", "ALL", "PATHS", "CONNECTED", "RELATIONSHIP",
                "MEMORY", "NODE", "EDGE", "GRAPH", "TEMPORAL", "RECENT", "BEFORE", "AFTER",
            ],
            functions: vec![
                "COUNT", "SUM", "AVG", "MIN", "MAX", "FIRST", "LAST", "COLLECT",
                "LENGTH", "SIZE", "TYPE", "ID", "PROPERTIES", "KEYS", "VALUES",
                "SUBSTRING", "LOWER", "UPPER", "TRIM", "REPLACE", "SPLIT", "CONCAT",
                "ROUND", "FLOOR", "CEIL", "ABS", "SQRT", "POW", "LOG", "EXP",
                "NOW", "DATE", "TIME", "DATETIME", "TIMESTAMP", "YEAR", "MONTH", "DAY",
                "HOUR", "MINUTE", "SECOND", "AGE", "DURATION", "FORMAT_DATE",
                "SIMILARITY", "DISTANCE", "CENTRALITY", "PAGERANK", "CLUSTERING",
                "SHORTEST_PATH", "ALL_PATHS", "CONNECTED_COMPONENTS", "DEGREE",
            ],
            _operators: vec![
                "+", "-", "*", "/", "%", "^", "=", "!=", "<>", "<", "<=", ">", ">=",
                "AND", "OR", "NOT", "LIKE", "REGEX", "IN", "BETWEEN", "EXISTS",
                "CONTAINS", "STARTS_WITH", "ENDS_WITH", "->", "<-", "<->", "~", "!~",
            ],
        })
    }

    /// Parse a SyQL query string into an AST
    pub fn parse(&self, query: &str) -> Result<Statement> {
        let mut lexer = Lexer::new(query);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        parser.parse_statement()
    }

    /// Get auto-completion suggestions for a partial query
    pub fn get_completions(&self, partial_query: &str, cursor_position: usize) -> Result<Vec<CompletionItem>> {
        let mut completions = Vec::new();
        
        // Get the word at cursor position
        let (word_start, current_word) = self.get_word_at_position(partial_query, cursor_position);
        
        // Analyze context to provide relevant completions
        let context = self.analyze_context(partial_query, word_start)?;
        
        match context {
            CompletionContext::Keyword => {
                for keyword in &self.keywords {
                    if keyword.to_lowercase().starts_with(&current_word.to_lowercase()) {
                        completions.push(CompletionItem {
                            text: keyword.to_string(),
                            item_type: CompletionType::Keyword,
                            description: format!("SyQL keyword: {}", keyword),
                            documentation: self.get_keyword_documentation(keyword),
                        });
                    }
                }
            },
            CompletionContext::Function => {
                for function in &self.functions {
                    if function.to_lowercase().starts_with(&current_word.to_lowercase()) {
                        completions.push(CompletionItem {
                            text: format!("{}()", function),
                            item_type: CompletionType::Function,
                            description: format!("SyQL function: {}", function),
                            documentation: self.get_function_documentation(function),
                        });
                    }
                }
            },
            CompletionContext::Property => {
                // Add common memory properties
                let properties = vec!["id", "content", "type", "created_at", "updated_at", "tags", "metadata"];
                for property in properties {
                    if property.starts_with(&current_word.to_lowercase()) {
                        completions.push(CompletionItem {
                            text: property.to_string(),
                            item_type: CompletionType::Property,
                            description: format!("Memory property: {}", property),
                            documentation: None,
                        });
                    }
                }
            },
            CompletionContext::MemoryType => {
                let types = vec!["text", "image", "audio", "video", "document", "code", "data"];
                for memory_type in types {
                    if memory_type.starts_with(&current_word.to_lowercase()) {
                        completions.push(CompletionItem {
                            text: memory_type.to_string(),
                            item_type: CompletionType::MemoryType,
                            description: format!("Memory type: {}", memory_type),
                            documentation: None,
                        });
                    }
                }
            },
            CompletionContext::RelationshipType => {
                let types = vec!["related_to", "contains", "references", "similar_to", "derived_from"];
                for rel_type in types {
                    if rel_type.starts_with(&current_word.to_lowercase()) {
                        completions.push(CompletionItem {
                            text: rel_type.to_string(),
                            item_type: CompletionType::RelationshipType,
                            description: format!("Relationship type: {}", rel_type),
                            documentation: None,
                        });
                    }
                }
            },

        }
        
        // Sort completions by relevance
        completions.sort_by(|a, b| {
            // Exact matches first
            let a_exact = a.text.to_lowercase() == current_word.to_lowercase();
            let b_exact = b.text.to_lowercase() == current_word.to_lowercase();
            
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.text.len().cmp(&b.text.len()),
            }
        });
        
        Ok(completions)
    }

    /// Get the word at a specific position in the query
    fn get_word_at_position(&self, query: &str, position: usize) -> (usize, String) {
        let chars: Vec<char> = query.chars().collect();
        let pos = position.min(chars.len());
        
        // Find word boundaries
        let mut start = pos;
        let mut end = pos;
        
        // Move start backward to find word start
        while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
            start -= 1;
        }
        
        // Move end forward to find word end
        while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
            end += 1;
        }
        
        let word: String = chars[start..end].iter().collect();
        (start, word)
    }

    /// Analyze context for completion
    fn analyze_context(&self, query: &str, position: usize) -> Result<CompletionContext> {
        let before_cursor = &query[..position];
        let tokens: Vec<&str> = before_cursor.split_whitespace().collect();
        
        if tokens.is_empty() {
            return Ok(CompletionContext::Keyword);
        }
        
        let last_token = tokens.last()
            .ok_or_else(|| MemoryError::InvalidQuery { message: "Empty token list".to_string() })?
            .to_uppercase();
        
        match last_token.as_str() {
            "SELECT" | "MATCH" | "WHERE" | "HAVING" | "ORDER" | "GROUP" => Ok(CompletionContext::Keyword),
            "FROM" | "JOIN" => Ok(CompletionContext::MemoryType),
            "TYPE" | "RELATIONSHIP" => Ok(CompletionContext::RelationshipType),
            "." => Ok(CompletionContext::Property),
            "(" => Ok(CompletionContext::Function),
            _ => {
                // Check if we're in a function context
                if before_cursor.contains('(') && !before_cursor.ends_with(')') {
                    Ok(CompletionContext::Function)
                } else if before_cursor.contains('.') {
                    Ok(CompletionContext::Property)
                } else {
                    Ok(CompletionContext::Keyword)
                }
            }
        }
    }

    /// Get documentation for a keyword
    fn get_keyword_documentation(&self, keyword: &str) -> Option<String> {
        match keyword {
            "SELECT" => Some("SELECT clause specifies which columns to return from the query".to_string()),
            "MATCH" => Some("MATCH clause specifies graph patterns to match against".to_string()),
            "WHERE" => Some("WHERE clause filters results based on conditions".to_string()),
            "ORDER" => Some("ORDER BY clause sorts results by specified expressions".to_string()),
            "LIMIT" => Some("LIMIT clause restricts the number of results returned".to_string()),
            "CREATE" => Some("CREATE statement creates new memories or relationships".to_string()),
            "UPDATE" => Some("UPDATE statement modifies existing memories or relationships".to_string()),
            "DELETE" => Some("DELETE statement removes memories or relationships".to_string()),
            _ => None,
        }
    }

    /// Get documentation for a function
    fn get_function_documentation(&self, function: &str) -> Option<String> {
        match function {
            "COUNT" => Some("COUNT() returns the number of rows or non-null values".to_string()),
            "SUM" => Some("SUM() returns the sum of numeric values".to_string()),
            "AVG" => Some("AVG() returns the average of numeric values".to_string()),
            "MIN" => Some("MIN() returns the minimum value".to_string()),
            "MAX" => Some("MAX() returns the maximum value".to_string()),
            "LENGTH" => Some("LENGTH() returns the length of a string or collection".to_string()),
            "SUBSTRING" => Some("SUBSTRING() extracts a portion of a string".to_string()),
            "NOW" => Some("NOW() returns the current timestamp".to_string()),
            "SIMILARITY" => Some("SIMILARITY() calculates similarity between memories".to_string()),
            "SHORTEST_PATH" => Some("SHORTEST_PATH() finds the shortest path between nodes".to_string()),
            _ => None,
        }
    }
}

/// Completion context types
#[derive(Debug, Clone)]
enum CompletionContext {
    Keyword,
    Function,
    Property,
    MemoryType,
    RelationshipType,

}

/// Token types for lexical analysis
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    
    // Identifiers and keywords
    Identifier(String),
    Keyword(String),
    
    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Not,
    Like,
    Regex,
    In,
    Between,
    
    // Punctuation
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Semicolon,
    Dot,
    Arrow,
    BackArrow,
    BiArrow,
    
    // Special
    Whitespace,
    Comment(String),
    EOF,
}

/// Token with position information
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub position: usize,
    pub length: usize,
}

/// Lexer for tokenizing SyQL queries
pub struct Lexer {
    input: String,
    position: usize,
    keywords: &'static [&'static str],
}

impl Lexer {
    /// Create a new lexer
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            position: 0,
            keywords: &[
                "SELECT", "FROM", "WHERE", "ORDER", "BY", "GROUP", "HAVING", "LIMIT",
                "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "INDEX",
                "MATCH", "RETURN", "WITH", "UNWIND", "MERGE", "OPTIONAL", "UNION",
                "AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "AS",
                "ASC", "DESC", "DISTINCT", "ALL", "ANY", "SOME", "EXISTS",
                "CASE", "WHEN", "THEN", "ELSE", "END", "IF", "COALESCE",
                "COUNT", "SUM", "AVG", "MIN", "MAX", "COLLECT", "DISTINCT",
                "STARTS", "ENDS", "CONTAINS", "REGEX", "SPLIT", "SUBSTRING",
                "TRIM", "LOWER", "UPPER", "REPLACE", "REVERSE", "SIZE", "LENGTH",
                "KEYS", "VALUES", "EXTRACT", "FILTER", "REDUCE", "RANGE",
                "RELATIONSHIPS", "NODES", "PATHS", "SHORTESTPATH", "ALLSHORTESTPATHS"
            ],
        }
    }

    /// Get current character
    fn current_char(&self) -> Option<char> {
        self.input.chars().nth(self.position)
    }



    /// Advance position by one character
    fn advance(&mut self) -> Option<char> {
        let ch = self.current_char();
        if ch.is_some() {
            self.position += 1;
        }
        ch
    }

    /// Check if we're at end of input
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Tokenize the input string
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        
        while let Some(token) = self.next_token()? {
            if !matches!(token.token_type, TokenType::Whitespace | TokenType::Comment(_)) {
                tokens.push(token);
            }
        }
        
        tokens.push(Token {
            token_type: TokenType::EOF,
            position: self.position,
            length: 0,
        });
        
        Ok(tokens)
    }

    /// Get the next token
    fn next_token(&mut self) -> Result<Option<Token>> {
        let start_position = self.position;

        if self.is_at_end() {
            return Ok(None);
        }

        match self.current_char() {
            None => Ok(None),
            Some(ch) => {
                match ch {
                    ' ' | '\t' | '\n' | '\r' => {
                        self.consume_whitespace();
                        Ok(Some(Token {
                            token_type: TokenType::Whitespace,
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    '-' if self.peek_ahead(1) == Some('-') => {
                        let comment = self.consume_line_comment();
                        Ok(Some(Token {
                            token_type: TokenType::Comment(comment),
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    '/' if self.peek_ahead(1) == Some('*') => {
                        let comment = self.consume_block_comment()?;
                        Ok(Some(Token {
                            token_type: TokenType::Comment(comment),
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    '\'' | '"' => {
                        let string_value = self.consume_string(ch)?;
                        Ok(Some(Token {
                            token_type: TokenType::String(string_value),
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    '0'..='9' => {
                        let number = self.consume_number()?;
                        Ok(Some(Token {
                            token_type: number,
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    'a'..='z' | 'A'..='Z' | '_' => {
                        let identifier = self.consume_identifier();
                        let token_type = if self.is_keyword(&identifier) {
                            TokenType::Keyword(identifier.to_uppercase())
                        } else {
                            match identifier.to_lowercase().as_str() {
                                "true" => TokenType::Boolean(true),
                                "false" => TokenType::Boolean(false),
                                "null" => TokenType::Null,
                                _ => TokenType::Identifier(identifier),
                            }
                        };
                        Ok(Some(Token {
                            token_type,
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    },
                    _ => {
                        let token_type = self.consume_operator_or_punctuation()?;
                        Ok(Some(Token {
                            token_type,
                            position: start_position,
                            length: self.position - start_position,
                        }))
                    }
                }
            }
        }
    }

    /// Consume whitespace characters
    fn consume_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Consume a line comment
    fn consume_line_comment(&mut self) -> String {
        let mut comment = String::new();

        // Skip the '--'
        self.advance();
        self.advance();

        while let Some(ch) = self.current_char() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            comment.push(ch);
            self.advance();
        }

        comment
    }

    /// Consume a block comment
    fn consume_block_comment(&mut self) -> Result<String> {
        let mut comment = String::new();

        // Skip the '/*'
        self.advance();
        self.advance();

        while let Some(ch) = self.advance() {
            if ch == '*' && self.current_char() == Some('/') {
                self.advance();
                break;
            }

            comment.push(ch);
        }

        Ok(comment)
    }

    /// Consume a string literal
    fn consume_string(&mut self, quote_char: char) -> Result<String> {
        let mut string_value = String::new();
        
        // Skip opening quote
        self.advance();

        while let Some(ch) = self.advance() {
            
            if ch == quote_char {
                break;
            }
            
            if ch == '\\' {
                // Handle escape sequences
                if let Some(escaped) = self.advance() {
                    match escaped {
                        'n' => string_value.push('\n'),
                        't' => string_value.push('\t'),
                        'r' => string_value.push('\r'),
                        '\\' => string_value.push('\\'),
                        '\'' => string_value.push('\''),
                        '"' => string_value.push('"'),
                        _ => {
                            string_value.push('\\');
                            string_value.push(escaped);
                        }
                    }
                }
            } else {
                string_value.push(ch);
            }
        }
        
        Ok(string_value)
    }

    /// Consume a number (integer or float)
    fn consume_number(&mut self) -> Result<TokenType> {
        let mut number_str = String::new();
        let mut is_float = false;

        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        if is_float {
            let value = number_str.parse::<f64>()
                .map_err(|_| MemoryError::InvalidQuery {
                    message: format!("Invalid float literal: {}", number_str),
                })?;
            Ok(TokenType::Float(value))
        } else {
            let value = number_str.parse::<i64>()
                .map_err(|_| MemoryError::InvalidQuery {
                    message: format!("Invalid integer literal: {}", number_str),
                })?;
            Ok(TokenType::Integer(value))
        }
    }

    /// Consume an identifier
    fn consume_identifier(&mut self) -> String {
        let mut identifier = String::new();

        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                identifier.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        identifier
    }

    /// Consume an operator or punctuation
    fn consume_operator_or_punctuation(&mut self) -> Result<TokenType> {
        let ch = self.advance().ok_or_else(|| crate::error::MemoryError::InvalidInput { message: "Unexpected end of input while parsing operator".to_string() })?;
        
        match ch {
            '+' => Ok(TokenType::Plus),
            '-' => {
                if self.current_char() == Some('>') {
                    self.advance();
                    Ok(TokenType::Arrow)
                } else {
                    Ok(TokenType::Minus)
                }
            },
            '*' => Ok(TokenType::Multiply),
            '/' => Ok(TokenType::Divide),
            '%' => Ok(TokenType::Modulo),
            '^' => Ok(TokenType::Power),
            '=' => Ok(TokenType::Equal),
            '!' => {
                if self.current_char() == Some('=') {
                    self.advance();
                    Ok(TokenType::NotEqual)
                } else {
                    Ok(TokenType::Not)
                }
            },
            '<' => {
                match self.current_char() {
                    Some('=') => {
                        self.advance();
                        Ok(TokenType::LessThanOrEqual)
                    },
                    Some('>') => {
                        self.advance();
                        Ok(TokenType::NotEqual)
                    },
                    Some('-') => {
                        self.advance();
                        if self.current_char() == Some('>') {
                            self.advance();
                            Ok(TokenType::BiArrow)
                        } else {
                            Ok(TokenType::BackArrow)
                        }
                    },
                    _ => Ok(TokenType::LessThan),
                }
            },
            '>' => {
                if self.current_char() == Some('=') {
                    self.advance();
                    Ok(TokenType::GreaterThanOrEqual)
                } else {
                    Ok(TokenType::GreaterThan)
                }
            },
            '(' => Ok(TokenType::LeftParen),
            ')' => Ok(TokenType::RightParen),
            '[' => Ok(TokenType::LeftBracket),
            ']' => Ok(TokenType::RightBracket),
            '{' => Ok(TokenType::LeftBrace),
            '}' => Ok(TokenType::RightBrace),
            ',' => Ok(TokenType::Comma),
            ';' => Ok(TokenType::Semicolon),
            '.' => Ok(TokenType::Dot),
            _ => Err(MemoryError::InvalidQuery {
                message: format!("Unexpected character: {}", ch),
            }),
        }
    }

    /// Peek ahead in the input
    fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.input.chars().nth(self.position + offset)
    }

    /// Check if an identifier is a keyword
    fn is_keyword(&self, identifier: &str) -> bool {
        self.keywords.iter().any(|&keyword| keyword.eq_ignore_ascii_case(identifier))
    }
}

/// Parser for building AST from tokens
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    /// Create a new parser
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Parse a statement
    pub fn parse_statement(&mut self) -> Result<Statement> {
        match self.current_token() {
            Some(Token { token_type: TokenType::Keyword(keyword), .. }) => {
                match keyword.as_str() {
                    "SELECT" | "MATCH" => {
                        let query = self.parse_query_statement()?;
                        Ok(Statement::Query(query))
                    },
                    "CREATE" => {
                        let create = self.parse_create_statement()?;
                        Ok(Statement::Create(create))
                    },
                    "UPDATE" => {
                        let update = self.parse_update_statement()?;
                        Ok(Statement::Update(update))
                    },
                    "DELETE" => {
                        let delete = self.parse_delete_statement()?;
                        Ok(Statement::Delete(delete))
                    },
                    "EXPLAIN" => {
                        self.advance(); // consume EXPLAIN
                        let statement = self.parse_statement()?;
                        Ok(Statement::Explain(Box::new(statement)))
                    },
                    "SHOW" => {
                        let show = self.parse_show_statement()?;
                        Ok(Statement::Show(show))
                    },
                    _ => Err(MemoryError::InvalidQuery {
                        message: format!("Unexpected keyword: {}", keyword),
                    }),
                }
            },
            _ => Err(MemoryError::InvalidQuery {
                message: "Expected statement keyword".to_string(),
            }),
        }
    }

    /// Get current token
    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    /// Advance to next token
    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    /// Check if current token matches the given type
    fn check(&self, token_type: &TokenType) -> bool {
        if let Some(token) = self.current_token() {
            std::mem::discriminant(&token.token_type) == std::mem::discriminant(token_type)
        } else {
            false
        }
    }

    /// Check if we're at the end of tokens
    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len() ||
        matches!(self.current_token(), Some(Token { token_type: TokenType::EOF, .. }))
    }

    /// Peek at current token
    fn peek(&self) -> Option<&Token> {
        self.current_token()
    }

    /// Match a specific keyword
    fn match_keyword(&mut self, keyword: &str) -> bool {
        if let Some(Token { token_type: TokenType::Keyword(k), .. }) = self.current_token() {
            if k.eq_ignore_ascii_case(keyword) {
                self.advance();
                return true;
            }
        }
        false
    }

    /// Match a specific token type
    fn match_token(&mut self, token_type: &TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Parse expression (placeholder)
    fn parse_expression(&mut self) -> Result<Expression> {
        // Simple expression parsing - just return a literal for now
        if let Some(token) = self.current_token().cloned() {
            match token.token_type {
                TokenType::String(s) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::String(s)))
                },
                TokenType::Integer(i) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Integer(i)))
                },
                TokenType::Float(f) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Float(f)))
                },
                TokenType::Boolean(b) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Boolean(b)))
                },
                TokenType::Null => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Null))
                },
                TokenType::Identifier(name) => {
                    self.advance();
                    Ok(Expression::Variable(name))
                },
                _ => Err(MemoryError::InvalidQuery {
                    message: "Expected expression".to_string(),
                }),
            }
        } else {
            Err(MemoryError::InvalidQuery {
                message: "Unexpected end of input".to_string(),
            })
        }
    }

    /// Parse query statement (placeholder implementation)
    fn parse_query_statement(&mut self) -> Result<QueryStatement> {
        // This is a simplified implementation
        // In a full implementation, this would parse the complete SELECT/MATCH syntax
        
        self.advance(); // consume SELECT/MATCH
        
        Ok(QueryStatement {
            select: SelectClause {
                distinct: false,
                expressions: vec![SelectExpression {
                    expression: Expression::Literal(Literal::String("*".to_string())),
                    alias: None,
                }],
            },
            from: FromClause::Memories {
                alias: None,
                filter: None,
            },
            where_clause: None,
            order_by: None,
            limit: None,
            group_by: None,
            having: None,
            with_clause: None,
        })
    }

    /// Parse create statement
    fn parse_create_statement(&mut self) -> Result<CreateStatement> {
        self.advance(); // consume CREATE

        // Expect MEMORY keyword
        if !self.match_keyword("MEMORY") {
            return Err(MemoryError::InvalidQuery {
                message: "Expected MEMORY after CREATE".to_string()
            });
        }

        let mut properties = HashMap::new();

        // Parse optional properties in parentheses
        if self.check(&TokenType::LeftParen) {
            self.advance(); // consume (

            while !self.check(&TokenType::RightParen) && !self.is_at_end() {
                // Parse property: key = value
                if let Some(Token { token_type: TokenType::Identifier(key), .. }) = self.peek() {
                    let key = key.clone();
                    self.advance();

                    if self.match_token(&TokenType::Equal) {
                        if let Some(Token { token_type: TokenType::String(value), .. }) = self.peek() {
                            let expr = Expression::Literal(Literal::String(value.clone()));
                            properties.insert(key, expr);
                            self.advance();
                        }
                    }
                }

                // Handle comma separation
                if self.check(&TokenType::Comma) {
                    self.advance();
                }
            }

            if !self.match_token(&TokenType::RightParen) {
                return Err(MemoryError::InvalidQuery {
                    message: "Expected ')' after properties".to_string()
                });
            }
        }

        Ok(CreateStatement::Memory { properties })
    }

    /// Parse update statement
    fn parse_update_statement(&mut self) -> Result<UpdateStatement> {
        self.advance(); // consume UPDATE

        // Parse target (MEMORIES or RELATIONSHIPS)
        let target = if self.match_keyword("MEMORIES") {
            if let Some(Token { token_type: TokenType::Identifier(pattern), .. }) = self.peek() {
                let pattern = pattern.clone();
                self.advance();
                UpdateTarget::Memories(pattern)
            } else {
                UpdateTarget::Memories("*".to_string())
            }
        } else if self.match_keyword("RELATIONSHIPS") {
            if let Some(Token { token_type: TokenType::Identifier(pattern), .. }) = self.peek() {
                let pattern = pattern.clone();
                self.advance();
                UpdateTarget::Relationships(pattern)
            } else {
                UpdateTarget::Relationships("*".to_string())
            }
        } else {
            return Err(MemoryError::InvalidQuery {
                message: "Expected MEMORIES or RELATIONSHIPS after UPDATE".to_string()
            });
        };

        // Parse SET clause
        if !self.match_keyword("SET") {
            return Err(MemoryError::InvalidQuery {
                message: "Expected SET clause in UPDATE statement".to_string()
            });
        }

        let mut set_clauses = Vec::new();

        loop {
            // Parse assignment: property = value
            if let Some(Token { token_type: TokenType::Identifier(property), .. }) = self.peek() {
                let property = property.clone();
                self.advance();

                if self.match_token(&TokenType::Equal) {
                    let value = self.parse_expression()?;
                    set_clauses.push(SetClause { property, value });
                }
            }

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }

        // Parse optional WHERE clause
        let where_clause = if self.match_keyword("WHERE") {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(UpdateStatement {
            target,
            set: set_clauses,
            where_clause,
        })
    }

    /// Parse delete statement
    fn parse_delete_statement(&mut self) -> Result<DeleteStatement> {
        self.advance(); // consume DELETE

        // Expect FROM keyword
        if !self.match_keyword("FROM") {
            return Err(MemoryError::InvalidQuery {
                message: "Expected FROM after DELETE".to_string()
            });
        }

        // Parse target (MEMORIES or RELATIONSHIPS)
        let target = if self.match_keyword("MEMORIES") {
            if let Some(Token { token_type: TokenType::Identifier(pattern), .. }) = self.peek() {
                let pattern = pattern.clone();
                self.advance();
                DeleteTarget::Memories(pattern)
            } else {
                DeleteTarget::Memories("*".to_string())
            }
        } else if self.match_keyword("RELATIONSHIPS") {
            if let Some(Token { token_type: TokenType::Identifier(pattern), .. }) = self.peek() {
                let pattern = pattern.clone();
                self.advance();
                DeleteTarget::Relationships(pattern)
            } else {
                DeleteTarget::Relationships("*".to_string())
            }
        } else {
            return Err(MemoryError::InvalidQuery {
                message: "Expected MEMORIES or RELATIONSHIPS after FROM".to_string()
            });
        };

        // Parse optional WHERE clause
        let where_clause = if self.match_keyword("WHERE") {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(DeleteStatement {
            target,
            where_clause,
        })
    }

    /// Parse show statement (placeholder)
    fn parse_show_statement(&mut self) -> Result<ShowStatement> {
        self.advance(); // consume SHOW
        
        Ok(ShowStatement::Memories)
    }
}
