//! SyQL Query Planner
//!
//! This module implements query planning for SyQL, converting optimized ASTs into
//! executable query plans with detailed execution strategies.

use super::ast::*;
use super::{QueryPlan, PlanNode, PlanNodeType, PlanStatistics, QueryValue};
use crate::error::Result;
use std::collections::HashMap;


/// Query planner for SyQL
pub struct QueryPlanner {
    /// Plan node counter for unique IDs
    node_counter: std::cell::RefCell<usize>,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new() -> Result<Self> {
        Ok(Self {
            node_counter: std::cell::RefCell::new(0),
        })
    }

    /// Create an execution plan from an optimized AST
    pub async fn create_plan(&self, statement: Statement) -> Result<QueryPlan> {
        match statement {
            Statement::Query(query) => self.create_query_plan(query).await,
            Statement::Create(create) => self.create_create_plan(create).await,
            Statement::Update(update) => self.create_update_plan(update).await,
            Statement::Delete(delete) => self.create_delete_plan(delete).await,
            Statement::Show(show) => self.create_show_plan(show).await,
            Statement::Explain(_inner) => {
                // Simplified EXPLAIN handling without recursion
                let node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::MemoryScan,
                    description: "Explain Plan".to_string(),
                    cost: 1.0,
                    rows: 1,
                    children: Vec::new(),
                    properties: HashMap::new(),
                };

                Ok(QueryPlan {
                    nodes: vec![node],
                    estimated_cost: 1.0,
                    estimated_rows: 1,
                    statistics: PlanStatistics {
                        total_nodes: 1,
                        scan_operations: 1,
                        join_operations: 0,
                        index_operations: 0,
                    },
                })
            },
        }
    }

    /// Create a query execution plan
    async fn create_query_plan(&self, query: QueryStatement) -> Result<QueryPlan> {
        let mut nodes = Vec::new();
        let mut estimated_cost = 0.0;
        // Create plan nodes for FROM clause
        let (from_nodes, from_cost, from_rows) = self.create_from_plan(&query.from).await?;
        nodes.extend(from_nodes);
        estimated_cost += from_cost;
        let mut estimated_rows = from_rows;

        // Create plan node for WHERE clause
        if let Some(where_expr) = &query.where_clause {
            let filter_node = self.create_filter_node(where_expr, estimated_rows).await?;
            estimated_cost += filter_node.cost;
            estimated_rows = (estimated_rows as f64 * 0.1) as usize; // Assume 10% selectivity
            nodes.push(filter_node);
        }

        // Create plan node for GROUP BY clause
        if let Some(group_by) = &query.group_by {
            let group_node = self.create_group_by_node(group_by, estimated_rows).await?;
            estimated_cost += group_node.cost;
            estimated_rows = (estimated_rows as f64 * 0.5) as usize; // Assume 50% reduction
            nodes.push(group_node);
        }

        // Create plan node for HAVING clause
        if let Some(having_expr) = &query.having {
            let having_node = self.create_having_node(having_expr, estimated_rows).await?;
            estimated_cost += having_node.cost;
            estimated_rows = (estimated_rows as f64 * 0.1) as usize; // Assume 10% selectivity
            nodes.push(having_node);
        }

        // Create plan node for ORDER BY clause
        if let Some(order_by) = &query.order_by {
            let sort_node = self.create_sort_node(order_by, estimated_rows).await?;
            estimated_cost += sort_node.cost;
            nodes.push(sort_node);
        }

        // Create plan node for SELECT clause (projection)
        let project_node = self.create_project_node(&query.select, estimated_rows).await?;
        estimated_cost += project_node.cost;
        nodes.push(project_node);

        // Create plan node for LIMIT clause
        if let Some(limit) = &query.limit {
            let limit_node = self.create_limit_node(limit, estimated_rows).await?;
            estimated_cost += limit_node.cost;
            estimated_rows = self.extract_limit_count(limit).min(estimated_rows);
            nodes.push(limit_node);
        }

        let statistics = PlanStatistics {
            total_nodes: nodes.len(),
            scan_operations: nodes.iter().filter(|n| matches!(n.node_type, PlanNodeType::MemoryScan | PlanNodeType::IndexScan)).count(),
            join_operations: nodes.iter().filter(|n| matches!(n.node_type, PlanNodeType::Join)).count(),
            index_operations: nodes.iter().filter(|n| matches!(n.node_type, PlanNodeType::IndexScan)).count(),
        };

        Ok(QueryPlan {
            nodes,
            estimated_cost,
            estimated_rows,
            statistics,
        })
    }

    /// Create plan for FROM clause
    async fn create_from_plan(&self, from: &FromClause) -> Result<(Vec<PlanNode>, f64, usize)> {
        match from {
            FromClause::Memories { alias, filter } => {
                let mut nodes = Vec::new();
                let mut cost = 0.0;
                let mut rows = 1000; // Default estimate

                // Create memory scan node
                let scan_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::MemoryScan,
                    description: format!("Memory Scan{}", 
                        if let Some(alias) = alias { format!(" AS {}", alias) } else { String::new() }
                    ),
                    cost: 100.0, // Base scan cost
                    rows,
                    children: Vec::new(),
                    properties: {
                        let mut props = HashMap::new();
                        if let Some(alias) = alias {
                            props.insert("alias".to_string(), QueryValue::String(alias.clone()));
                        }
                        props
                    },
                };
                cost += scan_node.cost;
                nodes.push(scan_node);

                // Add filter node if present
                if let Some(filter_expr) = filter {
                    let filter_node = self.create_filter_node(filter_expr, rows).await?;
                    cost += filter_node.cost;
                    rows = (rows as f64 * 0.1) as usize; // Assume 10% selectivity
                    nodes.push(filter_node);
                }

                Ok((nodes, cost, rows))
            },
            FromClause::Pattern { patterns } => {
                let mut nodes = Vec::new();
                let mut cost = 0.0;
                let rows = 500; // Pattern matching estimate

                for (i, pattern) in patterns.iter().enumerate() {
                    let pattern_node = PlanNode {
                        id: self.next_node_id(),
                        node_type: PlanNodeType::RelationshipTraversal,
                        description: format!("Pattern Match {}", i + 1),
                        cost: 50.0 * pattern.relationships.len() as f64,
                        rows,
                        children: Vec::new(),
                        properties: {
                            let mut props = HashMap::new();
                            props.insert("pattern_id".to_string(), QueryValue::Integer(i as i64));
                            props.insert("node_count".to_string(), QueryValue::Integer(pattern.nodes.len() as i64));
                            props.insert("relationship_count".to_string(), QueryValue::Integer(pattern.relationships.len() as i64));
                            props
                        },
                    };
                    cost += pattern_node.cost;
                    nodes.push(pattern_node);
                }

                Ok((nodes, cost, rows))
            },
            FromClause::Path { path_pattern } => {
                let path_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::PathExpansion,
                    description: format!("Path Finding ({:?})", path_pattern.algorithm),
                    cost: 200.0, // Path finding is expensive
                    rows: 100, // Paths are typically fewer
                    children: Vec::new(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("algorithm".to_string(), QueryValue::String(format!("{:?}", path_pattern.algorithm)));
                        if let Some(max_length) = path_pattern.max_length {
                            props.insert("max_length".to_string(), QueryValue::Integer(max_length as i64));
                        }
                        props
                    },
                };

                Ok((vec![path_node], 200.0, 100))
            },
            FromClause::Join { left: _, right: _, join_type, condition } => {
                // Simplified join handling without recursion
                let left_rows = 1000; // Default estimate
                let right_rows = 1000; // Default estimate
                let left_cost = 10.0;
                let right_cost = 10.0;

                let mut nodes = Vec::new();
                // Create simple scan nodes for left and right sides
                let left_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::MemoryScan,
                    description: "Left Join Input".to_string(),
                    cost: left_cost,
                    rows: left_rows,
                    children: Vec::new(),
                    properties: HashMap::new(),
                };
                let right_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::MemoryScan,
                    description: "Right Join Input".to_string(),
                    cost: right_cost,
                    rows: right_rows,
                    children: Vec::new(),
                    properties: HashMap::new(),
                };
                nodes.push(left_node);
                nodes.push(right_node);

                let join_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::Join,
                    description: format!("{:?} Join", join_type),
                    cost: (left_rows * right_rows) as f64 * 0.001, // Nested loop join cost
                    rows: match join_type {
                        JoinType::Inner => (left_rows * right_rows) / 10, // Assume 10% join selectivity
                        JoinType::Left => left_rows,
                        JoinType::Right => right_rows,
                        JoinType::Full => left_rows + right_rows,
                        JoinType::Cross => left_rows * right_rows,
                    },
                    children: Vec::new(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("join_type".to_string(), QueryValue::String(format!("{:?}", join_type)));
                        props.insert("left_rows".to_string(), QueryValue::Integer(left_rows as i64));
                        props.insert("right_rows".to_string(), QueryValue::Integer(right_rows as i64));
                        if condition.is_some() {
                            props.insert("has_condition".to_string(), QueryValue::Boolean(true));
                        }
                        props
                    },
                };

                let total_cost = left_cost + right_cost + join_node.cost;
                let total_rows = join_node.rows;
                nodes.push(join_node);

                Ok((nodes, total_cost, total_rows))
            },
            FromClause::Subquery { query: _, alias } => {
                // Simplified subquery handling without recursion
                let estimated_cost = 50.0; // Default subquery cost
                let estimated_rows = 100; // Default subquery rows
                
                let subquery_node = PlanNode {
                    id: self.next_node_id(),
                    node_type: PlanNodeType::MemoryScan, // Treat as a scan
                    description: format!("Subquery AS {}", alias),
                    cost: estimated_cost,
                    rows: estimated_rows,
                    children: Vec::new(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("alias".to_string(), QueryValue::String(alias.clone()));
                        props.insert("subquery_nodes".to_string(), QueryValue::Integer(1));
                        props
                    },
                };

                let nodes = vec![subquery_node];

                Ok((nodes, estimated_cost, estimated_rows))
            },
        }
    }

    /// Create filter node
    async fn create_filter_node(&self, expr: &Expression, input_rows: usize) -> Result<PlanNode> {
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Filter,
            description: "Filter".to_string(),
            cost: input_rows as f64 * 0.01, // Small cost per row
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("expression_type".to_string(), QueryValue::String(self.expression_type_name(expr)));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                props
            },
        })
    }

    /// Create GROUP BY node
    async fn create_group_by_node(&self, group_by: &GroupByClause, input_rows: usize) -> Result<PlanNode> {
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Aggregate,
            description: "Group By".to_string(),
            cost: input_rows as f64 * 0.1, // Grouping cost
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("group_expressions".to_string(), QueryValue::Integer(group_by.expressions.len() as i64));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                props
            },
        })
    }

    /// Create HAVING node
    async fn create_having_node(&self, expr: &Expression, input_rows: usize) -> Result<PlanNode> {
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Filter,
            description: "Having".to_string(),
            cost: input_rows as f64 * 0.01,
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("expression_type".to_string(), QueryValue::String(self.expression_type_name(expr)));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                props
            },
        })
    }

    /// Create ORDER BY node
    async fn create_sort_node(&self, order_by: &OrderByClause, input_rows: usize) -> Result<PlanNode> {
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Sort,
            description: "Sort".to_string(),
            cost: input_rows as f64 * (input_rows as f64).log2() * 0.001, // O(n log n) sort cost
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("sort_expressions".to_string(), QueryValue::Integer(order_by.expressions.len() as i64));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                props
            },
        })
    }

    /// Create projection node
    async fn create_project_node(&self, select: &SelectClause, input_rows: usize) -> Result<PlanNode> {
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Project,
            description: if select.distinct { "Project (Distinct)" } else { "Project" }.to_string(),
            cost: input_rows as f64 * 0.005, // Small projection cost
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("distinct".to_string(), QueryValue::Boolean(select.distinct));
                props.insert("expressions".to_string(), QueryValue::Integer(select.expressions.len() as i64));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                props
            },
        })
    }

    /// Create LIMIT node
    async fn create_limit_node(&self, limit: &LimitClause, input_rows: usize) -> Result<PlanNode> {
        let limit_count = self.extract_limit_count(limit);
        
        Ok(PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::Limit,
            description: "Limit".to_string(),
            cost: limit_count.min(input_rows) as f64 * 0.001, // Very small cost
            rows: input_rows,
            children: Vec::new(),
            properties: {
                let mut props = HashMap::new();
                props.insert("limit_count".to_string(), QueryValue::Integer(limit_count as i64));
                props.insert("input_rows".to_string(), QueryValue::Integer(input_rows as i64));
                if limit.offset.is_some() {
                    props.insert("has_offset".to_string(), QueryValue::Boolean(true));
                }
                props
            },
        })
    }

    /// Create plan for CREATE statement
    async fn create_create_plan(&self, _create: CreateStatement) -> Result<QueryPlan> {
        let node = PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::MemoryScan, // Placeholder
            description: "Create".to_string(),
            cost: 10.0,
            rows: 1,
            children: Vec::new(),
            properties: HashMap::new(),
        };

        Ok(QueryPlan {
            nodes: vec![node],
            estimated_cost: 10.0,
            estimated_rows: 1,
            statistics: PlanStatistics {
                total_nodes: 1,
                scan_operations: 0,
                join_operations: 0,
                index_operations: 0,
            },
        })
    }

    /// Create plan for UPDATE statement
    async fn create_update_plan(&self, _update: UpdateStatement) -> Result<QueryPlan> {
        let node = PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::MemoryScan, // Placeholder
            description: "Update".to_string(),
            cost: 20.0,
            rows: 1,
            children: Vec::new(),
            properties: HashMap::new(),
        };

        Ok(QueryPlan {
            nodes: vec![node],
            estimated_cost: 20.0,
            estimated_rows: 1,
            statistics: PlanStatistics {
                total_nodes: 1,
                scan_operations: 0,
                join_operations: 0,
                index_operations: 0,
            },
        })
    }

    /// Create plan for DELETE statement
    async fn create_delete_plan(&self, _delete: DeleteStatement) -> Result<QueryPlan> {
        let node = PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::MemoryScan, // Placeholder
            description: "Delete".to_string(),
            cost: 15.0,
            rows: 1,
            children: Vec::new(),
            properties: HashMap::new(),
        };

        Ok(QueryPlan {
            nodes: vec![node],
            estimated_cost: 15.0,
            estimated_rows: 1,
            statistics: PlanStatistics {
                total_nodes: 1,
                scan_operations: 0,
                join_operations: 0,
                index_operations: 0,
            },
        })
    }

    /// Create plan for SHOW statement
    async fn create_show_plan(&self, show: ShowStatement) -> Result<QueryPlan> {
        let description = match show {
            ShowStatement::Memories => "Show Memories",
            ShowStatement::Relationships => "Show Relationships",
            ShowStatement::Indexes => "Show Indexes",
            ShowStatement::Statistics => "Show Statistics",
            ShowStatement::Schema => "Show Schema",
        };

        let node = PlanNode {
            id: self.next_node_id(),
            node_type: PlanNodeType::MemoryScan,
            description: description.to_string(),
            cost: 5.0,
            rows: 100, // Typical metadata size
            children: Vec::new(),
            properties: HashMap::new(),
        };

        Ok(QueryPlan {
            nodes: vec![node],
            estimated_cost: 5.0,
            estimated_rows: 100,
            statistics: PlanStatistics {
                total_nodes: 1,
                scan_operations: 1,
                join_operations: 0,
                index_operations: 0,
            },
        })
    }

    /// Generate next node ID
    fn next_node_id(&self) -> String {
        let mut counter = self.node_counter.borrow_mut();
        *counter += 1;
        format!("node_{}", *counter)
    }

    /// Get expression type name for debugging
    fn expression_type_name(&self, expr: &Expression) -> String {
        match expr {
            Expression::Literal(_) => "Literal",
            Expression::Variable(_) => "Variable",
            Expression::Property { .. } => "Property",
            Expression::Function { .. } => "Function",
            Expression::Binary { .. } => "Binary",
            Expression::Unary { .. } => "Unary",
            Expression::Case { .. } => "Case",
            Expression::In { .. } => "In",
            Expression::Between { .. } => "Between",
            Expression::Exists { .. } => "Exists",
            Expression::Subquery(_) => "Subquery",
            Expression::Array(_) => "Array",
            Expression::Map(_) => "Map",
        }.to_string()
    }

    /// Extract limit count from limit clause
    fn extract_limit_count(&self, limit: &LimitClause) -> usize {
        match &limit.count {
            Expression::Literal(Literal::Integer(n)) => *n as usize,
            _ => 100, // Default limit
        }
    }
}
