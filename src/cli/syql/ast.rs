//! SyQL Abstract Syntax Tree (AST) Definitions
//!
//! This module defines the AST nodes for the SyQL query language, providing a structured
//! representation of parsed queries that can be optimized and executed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Root AST node for SyQL queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    /// SELECT/MATCH query
    Query(QueryStatement),
    /// CREATE statement
    Create(CreateStatement),
    /// UPDATE statement
    Update(UpdateStatement),
    /// DELETE statement
    Delete(DeleteStatement),
    /// EXPLAIN statement
    Explain(Box<Statement>),
    /// SHOW statement
    Show(ShowStatement),
}

/// Query statement (SELECT/MATCH)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatement {
    /// SELECT clause
    pub select: SelectClause,
    /// FROM/MATCH clause
    pub from: FromClause,
    /// WHERE clause
    pub where_clause: Option<Expression>,
    /// ORDER BY clause
    pub order_by: Option<OrderByClause>,
    /// LIMIT clause
    pub limit: Option<LimitClause>,
    /// GROUP BY clause
    pub group_by: Option<GroupByClause>,
    /// HAVING clause
    pub having: Option<Expression>,
    /// WITH clause (for CTEs)
    pub with_clause: Option<WithClause>,
}

/// SELECT clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectClause {
    /// Whether DISTINCT is specified
    pub distinct: bool,
    /// Selected expressions
    pub expressions: Vec<SelectExpression>,
}

/// Select expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectExpression {
    /// Expression to select
    pub expression: Expression,
    /// Optional alias
    pub alias: Option<String>,
}

/// FROM/MATCH clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FromClause {
    /// Memory collection scan
    Memories {
        alias: Option<String>,
        filter: Option<Expression>,
    },
    /// Graph pattern matching
    Pattern {
        patterns: Vec<GraphPattern>,
    },
    /// Path queries
    Path {
        path_pattern: PathPattern,
    },
    /// Subquery
    Subquery {
        query: Box<QueryStatement>,
        alias: String,
    },
    /// Join
    Join {
        left: Box<FromClause>,
        right: Box<FromClause>,
        join_type: JoinType,
        condition: Option<Expression>,
    },
}

/// Graph pattern for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Node patterns
    pub nodes: Vec<NodePattern>,
    /// Relationship patterns
    pub relationships: Vec<RelationshipPattern>,
    /// Optional patterns
    pub optional: bool,
}

/// Node pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePattern {
    /// Variable name
    pub variable: String,
    /// Node labels/types
    pub labels: Vec<String>,
    /// Property constraints
    pub properties: HashMap<String, Expression>,
    /// Optional node
    pub optional: bool,
}

/// Relationship pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipPattern {
    /// Variable name
    pub variable: Option<String>,
    /// From node variable
    pub from_node: String,
    /// To node variable
    pub to_node: String,
    /// Relationship types
    pub types: Vec<String>,
    /// Direction
    pub direction: RelationshipDirection,
    /// Property constraints
    pub properties: HashMap<String, Expression>,
    /// Length constraints (for variable-length paths)
    pub length: Option<LengthConstraint>,
}

/// Relationship direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipDirection {
    Outgoing,
    Incoming,
    Bidirectional,
}

/// Length constraint for variable-length paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthConstraint {
    /// Minimum length
    pub min: Option<usize>,
    /// Maximum length
    pub max: Option<usize>,
}

/// Path pattern for path queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathPattern {
    /// Start node
    pub start: NodePattern,
    /// End node
    pub end: NodePattern,
    /// Path constraints
    pub constraints: Vec<PathConstraint>,
    /// Maximum path length
    pub max_length: Option<usize>,
    /// Path finding algorithm
    pub algorithm: PathAlgorithm,
}

/// Path constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathConstraint {
    /// Avoid nodes
    AvoidNodes(Vec<String>),
    /// Avoid relationships
    AvoidRelationships(Vec<String>),
    /// Required nodes
    RequiredNodes(Vec<String>),
    /// Required relationships
    RequiredRelationships(Vec<String>),
    /// Weight constraint
    WeightConstraint(Expression),
}

/// Path finding algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathAlgorithm {
    Shortest,
    AllPaths,
    Dijkstra,
    AStar,
    BellmanFord,
}

/// Join types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// ORDER BY clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderByClause {
    /// Order expressions
    pub expressions: Vec<OrderExpression>,
}

/// Order expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderExpression {
    /// Expression to order by
    pub expression: Expression,
    /// Order direction
    pub direction: OrderDirection,
    /// Null handling
    pub nulls: NullsOrder,
}

/// Order direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderDirection {
    Ascending,
    Descending,
}

/// Nulls order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NullsOrder {
    First,
    Last,
    Default,
}

/// LIMIT clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitClause {
    /// Number of rows to limit
    pub count: Expression,
    /// Offset
    pub offset: Option<Expression>,
}

/// GROUP BY clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupByClause {
    /// Grouping expressions
    pub expressions: Vec<Expression>,
}

/// WITH clause for Common Table Expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithClause {
    /// CTE definitions
    pub ctes: Vec<CommonTableExpression>,
}

/// Common Table Expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonTableExpression {
    /// CTE name
    pub name: String,
    /// Column names
    pub columns: Option<Vec<String>>,
    /// CTE query
    pub query: Box<QueryStatement>,
}

/// CREATE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreateStatement {
    /// Create memory
    Memory {
        properties: HashMap<String, Expression>,
    },
    /// Create relationship
    Relationship {
        from_node: Expression,
        to_node: Expression,
        relationship_type: String,
        properties: HashMap<String, Expression>,
    },
    /// Create index
    Index {
        name: String,
        on: IndexTarget,
        properties: Vec<String>,
    },
}

/// Index target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexTarget {
    Memories,
    Relationships,
}

/// UPDATE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatement {
    /// Target to update
    pub target: UpdateTarget,
    /// SET clause
    pub set: Vec<SetClause>,
    /// WHERE clause
    pub where_clause: Option<Expression>,
}

/// Update target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateTarget {
    Memories(String),
    Relationships(String),
}

/// SET clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetClause {
    /// Property to set
    pub property: String,
    /// Value expression
    pub value: Expression,
}

/// DELETE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteStatement {
    /// Target to delete
    pub target: DeleteTarget,
    /// WHERE clause
    pub where_clause: Option<Expression>,
}

/// Delete target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeleteTarget {
    Memories(String),
    Relationships(String),
}

/// SHOW statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShowStatement {
    /// Show memories
    Memories,
    /// Show relationships
    Relationships,
    /// Show indexes
    Indexes,
    /// Show statistics
    Statistics,
    /// Show schema
    Schema,
}

/// Expression AST node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Literal value
    Literal(Literal),
    /// Variable reference
    Variable(String),
    /// Property access
    Property {
        object: Box<Expression>,
        property: String,
    },
    /// Function call
    Function {
        name: String,
        args: Vec<Expression>,
    },
    /// Binary operation
    Binary {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    /// Unary operation
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    /// CASE expression
    Case {
        when_clauses: Vec<WhenClause>,
        else_clause: Option<Box<Expression>>,
    },
    /// IN expression
    In {
        expression: Box<Expression>,
        list: Vec<Expression>,
    },
    /// BETWEEN expression
    Between {
        expression: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
    },
    /// EXISTS expression
    Exists {
        subquery: Box<QueryStatement>,
    },
    /// Subquery expression
    Subquery(Box<QueryStatement>),
    /// Array/List expression
    Array(Vec<Expression>),
    /// Map/Object expression
    Map(HashMap<String, Expression>),
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    DateTime(DateTime<Utc>),
}

/// Binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    
    // Logical
    And,
    Or,
    
    // String
    Like,
    NotLike,
    Regex,
    NotRegex,
    
    // Array/List
    Contains,
    NotContains,
    
    // Graph-specific
    Connected,
    NotConnected,
    PathExists,
}

/// Unary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
    Plus,
    IsNull,
    IsNotNull,
}

/// WHEN clause for CASE expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhenClause {
    /// Condition
    pub condition: Expression,
    /// Result
    pub result: Expression,
}
