//! Tests for improved search scoring algorithms

#[cfg(all(test, feature = "search-scoring-tests"))]
mod search_scoring_tests {
    use synaptic::memory::management::search::AdvancedSearchEngine;
    use synaptic::memory::types::{MemoryEntry, MemoryType};

    fn create_test_search_engine() -> AdvancedSearchEngine {
        AdvancedSearchEngine::new()
    }

    fn create_test_memory(content: &str, tags: Vec<String>) -> MemoryEntry {
        let mut memory = MemoryEntry::new(
            "test_key".to_string(),
            content.to_string(),
            MemoryType::ShortTerm
        );
        memory.metadata.tags = tags;
        memory.metadata.importance = 0.7;
        memory.metadata.confidence = 0.8;
        memory
    }

    #[test]
    fn test_sophisticated_content_type_scoring() {
        let engine = create_test_search_engine();

        // Test URL content (should get lower score)
        let url_content = "Check out this link: https://example.com/article and www.github.com";
        let url_score = engine.calculate_sophisticated_content_type_score(url_content);
        
        // Test code content (should get high score)
        let code_content = "function calculateSum(a, b) { return a + b; } class MyClass { def method(): pass }";
        let code_score = engine.calculate_sophisticated_content_type_score(code_content);
        
        // Test documentation content (should get highest score)
        let doc_content = "This is a comprehensive explanation of the algorithm. The overview shows how to implement the steps in the tutorial guide.";
        let doc_score = engine.calculate_sophisticated_content_type_score(doc_content);
        
        // Test structured content (should get high score)
        let structured_content = "Steps to follow:\n- First step\n- Second step\n1. Initialize\n2. Process\n3. Finalize\n```code block```";
        let structured_score = engine.calculate_sophisticated_content_type_score(structured_content);
        
        // Test actionable content (should get high score)
        let actionable_content = "Action plan: implement the strategy to execute the task and achieve the goal through planned objectives.";
        let actionable_score = engine.calculate_sophisticated_content_type_score(actionable_content);
        
        // Test knowledge content (should get highest score)
        let knowledge_content = "The concept behind this theory is based on the principle that understanding the definition provides insight into the meaning.";
        let knowledge_score = engine.calculate_sophisticated_content_type_score(knowledge_content);

        // Verify scoring hierarchy
        assert!(doc_score > url_score, "Documentation should score higher than URLs");
        assert!(code_score > url_score, "Code should score higher than URLs");
        assert!(structured_score > url_score, "Structured content should score higher than URLs");
        assert!(actionable_score > url_score, "Actionable content should score higher than URLs");
        assert!(knowledge_score > url_score, "Knowledge content should score higher than URLs");
        
        // Documentation and knowledge should be among the highest
        assert!(doc_score >= 0.8, "Documentation should get high score");
        assert!(knowledge_score >= 0.8, "Knowledge content should get high score");
        
        // URL content should get moderate score
        assert!(url_score >= 0.5 && url_score <= 0.7, "URL content should get moderate score");
    }

    #[test]
    fn test_no_todo_fixme_dependency() {
        let engine = create_test_search_engine();

        // Test content with TODO/FIXME - should not get artificially high scores
        let todo_content = "TODO: implement this feature and FIXME: resolve this bug";
        let todo_score = engine.calculate_sophisticated_content_type_score(todo_content);
        
        // Test equivalent content without TODO/FIXME
        let regular_content = "Need to implement this feature and resolve this bug";
        let regular_score = engine.calculate_sophisticated_content_type_score(regular_content);
        
        // Scores should be similar (not artificially boosted by TODO/FIXME)
        let score_diff = (todo_score - regular_score).abs();
        assert!(score_diff < 0.1, "TODO/FIXME should not artificially boost scores");
        
        // Both should get reasonable scores based on content quality
        assert!(todo_score >= 0.5, "Content with TODO should still get reasonable score");
        assert!(regular_score >= 0.5, "Regular content should get reasonable score");
    }

    #[test]
    fn test_content_length_scoring() {
        let engine = create_test_search_engine();

        // Test very short content
        let short_content = "Short";
        let short_score = engine.calculate_sophisticated_content_type_score(short_content);
        
        // Test medium content
        let medium_content = "This is a medium-length piece of content that provides some useful information about a topic. ".repeat(3);
        let medium_score = engine.calculate_sophisticated_content_type_score(&medium_content);
        
        // Test long content
        let long_content = "This is a comprehensive piece of content that provides detailed information about a complex topic. ".repeat(15);
        let long_score = engine.calculate_sophisticated_content_type_score(&long_content);

        // Longer content should generally score higher
        assert!(medium_score > short_score, "Medium content should score higher than short");
        assert!(long_score > medium_score, "Long content should score higher than medium");
        
        // But all should be within reasonable bounds
        assert!(short_score >= 0.3, "Even short content should get some score");
        assert!(long_score <= 1.0, "Scores should not exceed 1.0");
    }

    #[test]
    fn test_multi_factor_scoring() {
        let engine = create_test_search_engine();

        // Test content that combines multiple positive factors
        let rich_content = r#"
        # Algorithm Implementation Guide
        
        This explanation provides a comprehensive overview of the implementation strategy.
        
        ## Steps to follow:
        - Initialize the data structures
        - Implement the core algorithm
        - Execute the optimization plan
        
        ```python
        def calculate_score(data):
            return sum(data) / len(data)
        ```
        
        The concept behind this approach is based on the principle of efficiency.
        This tutorial demonstrates how to achieve the objective through understanding.
        "#;
        
        let rich_score = engine.calculate_sophisticated_content_type_score(rich_content);
        
        // Test content with fewer positive factors
        let simple_content = "Just a simple note about something.";
        let simple_score = engine.calculate_sophisticated_content_type_score(simple_content);
        
        // Rich content should score significantly higher
        assert!(rich_score > simple_score, "Rich content should score higher than simple content");
        assert!(rich_score >= 0.8, "Rich content should get high score");
        assert!(simple_score >= 0.4, "Simple content should still get reasonable score");
    }

    #[test]
    fn test_code_content_detection() {
        let engine = create_test_search_engine();

        // Test various code patterns
        let python_code = "def function_name(): return value if condition else None";
        let javascript_code = "function myFunc() { const result = data.map(item => item.value); }";
        let rust_code = "fn calculate() -> Result<i32> { let mut sum = 0; for i in 0..10 { sum += i; } }";
        let sql_code = "SELECT * FROM table WHERE condition = true";
        
        let python_score = engine.calculate_sophisticated_content_type_score(python_code);
        let js_score = engine.calculate_sophisticated_content_type_score(javascript_code);
        let rust_score = engine.calculate_sophisticated_content_type_score(rust_code);
        let sql_score = engine.calculate_sophisticated_content_type_score(sql_code);
        
        // All code content should get good scores
        assert!(python_score >= 0.7, "Python code should get good score");
        assert!(js_score >= 0.7, "JavaScript code should get good score");
        assert!(rust_score >= 0.7, "Rust code should get good score");
        assert!(sql_score >= 0.6, "SQL code should get reasonable score");
    }

    #[test]
    fn test_documentation_content_detection() {
        let engine = create_test_search_engine();

        // Test documentation patterns
        let api_doc = "This function explanation describes the parameters and return values. The overview shows usage examples.";
        let tutorial = "Step-by-step guide on how to implement the feature. This tutorial covers all the basics.";
        let summary = "Summary of the key concepts and principles. This description provides comprehensive understanding.";
        
        let api_score = engine.calculate_sophisticated_content_type_score(api_doc);
        let tutorial_score = engine.calculate_sophisticated_content_type_score(tutorial);
        let summary_score = engine.calculate_sophisticated_content_type_score(summary);
        
        // Documentation should get high scores
        assert!(api_score >= 0.8, "API documentation should get high score");
        assert!(tutorial_score >= 0.8, "Tutorial content should get high score");
        assert!(summary_score >= 0.8, "Summary content should get high score");
    }

    #[test]
    fn test_actionable_content_detection() {
        let engine = create_test_search_engine();

        // Test actionable content patterns
        let action_plan = "Action items: implement the new strategy, execute the plan, and achieve the objectives through coordinated tasks.";
        let goals = "Goals for this quarter: complete the project, implement improvements, and execute the strategy.";
        let tasks = "Task list: analyze requirements, plan implementation, execute development, achieve milestones.";
        
        let action_score = engine.calculate_sophisticated_content_type_score(action_plan);
        let goals_score = engine.calculate_sophisticated_content_type_score(goals);
        let tasks_score = engine.calculate_sophisticated_content_type_score(tasks);
        
        // Actionable content should get high scores
        assert!(action_score >= 0.8, "Action plans should get high score");
        assert!(goals_score >= 0.8, "Goals should get high score");
        assert!(tasks_score >= 0.8, "Task lists should get high score");
    }

    #[test]
    fn test_knowledge_content_detection() {
        let engine = create_test_search_engine();

        // Test knowledge content patterns
        let concepts = "The concept of machine learning is based on the principle that algorithms can learn from data to gain understanding.";
        let theory = "This theory explains the fundamental principles behind the phenomenon, providing insight into the underlying meaning.";
        let definitions = "Definition: A data structure is a concept that organizes information. Understanding this principle provides insight.";
        
        let concepts_score = engine.calculate_sophisticated_content_type_score(concepts);
        let theory_score = engine.calculate_sophisticated_content_type_score(theory);
        let definitions_score = engine.calculate_sophisticated_content_type_score(definitions);
        
        // Knowledge content should get very high scores
        assert!(concepts_score >= 0.85, "Conceptual content should get very high score");
        assert!(theory_score >= 0.85, "Theoretical content should get very high score");
        assert!(definitions_score >= 0.85, "Definitions should get very high score");
    }

    #[test]
    fn test_structured_content_detection() {
        let engine = create_test_search_engine();

        // Test structured content patterns
        let bullet_list = "Key points:\n- First important point\n- Second critical aspect\n- Third essential element";
        let numbered_list = "Process steps:\n1. Initialize the system\n2. Configure parameters\n3. Execute the workflow";
        let mixed_structure = "Overview:\n- Main concept\n1. First step\n2. Second step\n```code example```\nConclusion: summary";
        
        let bullet_score = engine.calculate_sophisticated_content_type_score(bullet_list);
        let numbered_score = engine.calculate_sophisticated_content_type_score(numbered_list);
        let mixed_score = engine.calculate_sophisticated_content_type_score(mixed_structure);
        
        // Structured content should get good scores
        assert!(bullet_score >= 0.7, "Bullet lists should get good score");
        assert!(numbered_score >= 0.7, "Numbered lists should get good score");
        assert!(mixed_score >= 0.8, "Mixed structure should get high score");
    }

    #[test]
    fn test_score_bounds_and_consistency() {
        let engine = create_test_search_engine();

        let test_contents = vec![
            "",
            "a",
            "Short content",
            "Medium length content with some useful information",
            "Very long content that goes on and on with lots of details and comprehensive information about various topics and concepts",
            "TODO: fix this FIXME: update that",
            "https://example.com www.test.com",
            "function test() { return true; }",
            "This explanation provides a comprehensive overview and tutorial guide",
        ];

        for content in test_contents {
            let score = engine.calculate_sophisticated_content_type_score(content);
            
            // All scores should be within valid bounds
            assert!(score >= 0.0, "Score should not be negative for content: '{}'", content);
            assert!(score <= 1.0, "Score should not exceed 1.0 for content: '{}'", content);
            
            // Non-empty content should get some score
            if !content.is_empty() {
                assert!(score > 0.0, "Non-empty content should get positive score: '{}'", content);
            }
        }
    }
}
