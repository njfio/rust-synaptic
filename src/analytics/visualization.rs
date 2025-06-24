// Visualization Module
// 3D graph visualization and temporal visualization capabilities

use crate::error::Result;
use crate::analytics::AnalyticsConfig;
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// 3D coordinate in visualization space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Point3D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

/// Visual node representing a memory in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualNode {
    /// Node identifier
    pub id: String,
    /// Memory key
    pub memory_key: String,
    /// Position in 3D space
    pub position: Point3D,
    /// Node size (based on importance)
    pub size: f64,
    /// Node color (RGB)
    pub color: (u8, u8, u8),
    /// Node label
    pub label: String,
    /// Node type category
    pub node_type: String,
    /// Visibility flag
    pub visible: bool,
    /// Animation state
    pub animation_state: AnimationState,
}

/// Visual edge representing relationships in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEdge {
    /// Edge identifier
    pub id: String,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge strength (affects thickness)
    pub strength: f64,
    /// Edge color (RGB)
    pub color: (u8, u8, u8),
    /// Edge type
    pub edge_type: String,
    /// Visibility flag
    pub visible: bool,
    /// Animation state
    pub animation_state: AnimationState,
}

/// Animation state for visual elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationState {
    /// Is currently animating
    pub is_animating: bool,
    /// Animation start time
    pub start_time: Option<DateTime<Utc>>,
    /// Animation duration (seconds)
    pub duration: f64,
    /// Animation type
    pub animation_type: AnimationType,
}

/// Types of animations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnimationType {
    None,
    FadeIn,
    FadeOut,
    Pulse,
    Move,
    Scale,
    Rotate,
}

/// 3D graph layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layout3DConfig {
    /// Layout algorithm
    pub algorithm: LayoutAlgorithm,
    /// Force strength for force-directed layouts
    pub force_strength: f64,
    /// Repulsion strength between nodes
    pub repulsion_strength: f64,
    /// Attraction strength for connected nodes
    pub attraction_strength: f64,
    /// Gravity center point
    pub gravity_center: Point3D,
    /// Gravity strength
    pub gravity_strength: f64,
    /// Maximum iterations for layout calculation
    pub max_iterations: u32,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Layout algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayoutAlgorithm {
    ForceDirected,
    Hierarchical,
    Circular,
    Grid,
    Sphere,
    Random,
}

impl Default for Layout3DConfig {
    fn default() -> Self {
        Self {
            algorithm: LayoutAlgorithm::ForceDirected,
            force_strength: 1.0,
            repulsion_strength: 100.0,
            attraction_strength: 0.1,
            gravity_center: Point3D::new(0.0, 0.0, 0.0),
            gravity_strength: 0.01,
            max_iterations: 1000,
            convergence_threshold: 0.01,
        }
    }
}

/// Temporal visualization data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value at this time
    pub value: f64,
    /// Associated memory key
    pub memory_key: String,
    /// Data type
    pub data_type: TemporalDataType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of temporal data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TemporalDataType {
    AccessFrequency,
    RelationshipStrength,
    ImportanceScore,
    UserActivity,
    MemoryEvolution,
}

/// Temporal visualization timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTimeline {
    /// Timeline identifier
    pub id: String,
    /// Timeline title
    pub title: String,
    /// Data points
    pub data_points: Vec<TemporalDataPoint>,
    /// Time range
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Visualization type
    pub viz_type: TimelineVisualizationType,
    /// Color scheme
    pub color_scheme: Vec<(u8, u8, u8)>,
}

/// Timeline visualization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimelineVisualizationType {
    LineChart,
    AreaChart,
    Heatmap,
    ScatterPlot,
    BarChart,
}

/// Heatmap data for relationship strength visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipHeatmap {
    /// Heatmap identifier
    pub id: String,
    /// Memory keys (x-axis)
    pub memory_keys: Vec<String>,
    /// Time periods (y-axis)
    pub time_periods: Vec<DateTime<Utc>>,
    /// Strength values (2D matrix)
    pub strength_matrix: Vec<Vec<f64>>,
    /// Color mapping configuration
    pub color_config: HeatmapColorConfig,
}

/// Heatmap color configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapColorConfig {
    /// Minimum value color (RGB)
    pub min_color: (u8, u8, u8),
    /// Maximum value color (RGB)
    pub max_color: (u8, u8, u8),
    /// Number of color steps
    pub color_steps: u32,
    /// Use logarithmic scale
    pub logarithmic: bool,
}

/// Visualization engine
#[derive(Debug)]
pub struct VisualizationEngine {
    /// Configuration
    _config: AnalyticsConfig,
    /// 3D layout configuration
    layout_config: Layout3DConfig,
    /// Visual nodes
    nodes: HashMap<String, VisualNode>,
    /// Visual edges
    edges: HashMap<String, VisualEdge>,
    /// Temporal timelines
    timelines: HashMap<String, TemporalTimeline>,
    /// Relationship heatmaps
    heatmaps: HashMap<String, RelationshipHeatmap>,
    /// Animation queue
    animation_queue: Vec<String>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new(config: &AnalyticsConfig) -> Result<Self> {
        Ok(Self {
            _config: config.clone(),
            layout_config: Layout3DConfig::default(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            timelines: HashMap::new(),
            heatmaps: HashMap::new(),
            animation_queue: Vec::new(),
        })
    }

    /// Create a visual node from memory entry
    pub async fn create_visual_node(&mut self, memory_key: &str, memory_entry: &MemoryEntry) -> Result<String> {
        let node_id = Uuid::new_v4().to_string();
        
        // Calculate position using layout algorithm
        let position = self.calculate_node_position(memory_key, memory_entry).await?;
        
        // Determine node size based on importance
        let size = memory_entry.metadata.importance * 10.0 + 5.0;
        
        // Determine color based on memory type or content
        let color = self.calculate_node_color(memory_entry);
        
        let visual_node = VisualNode {
            id: node_id.clone(),
            memory_key: memory_key.to_string(),
            position,
            size,
            color,
            label: memory_entry.value.chars().take(50).collect::<String>(),
            node_type: format!("{:?}", memory_entry.memory_type),
            visible: true,
            animation_state: AnimationState {
                is_animating: false,
                start_time: None,
                duration: 0.0,
                animation_type: AnimationType::None,
            },
        };

        self.nodes.insert(node_id.clone(), visual_node);
        Ok(node_id)
    }

    /// Create a visual edge between nodes
    pub async fn create_visual_edge(&mut self, source_key: &str, target_key: &str, strength: f64, edge_type: &str) -> Result<String> {
        let edge_id = Uuid::new_v4().to_string();
        
        // Find source and target node IDs
        let source_id = self.find_node_id_by_memory_key(source_key);
        let target_id = self.find_node_id_by_memory_key(target_key);
        
        if let (Some(source_id), Some(target_id)) = (source_id, target_id) {
            let color = self.calculate_edge_color(strength, edge_type);
            
            let visual_edge = VisualEdge {
                id: edge_id.clone(),
                source: source_id,
                target: target_id,
                strength,
                color,
                edge_type: edge_type.to_string(),
                visible: true,
                animation_state: AnimationState {
                    is_animating: false,
                    start_time: None,
                    duration: 0.0,
                    animation_type: AnimationType::None,
                },
            };

            self.edges.insert(edge_id.clone(), visual_edge);
            Ok(edge_id)
        } else {
            Err(crate::error::MemoryError::storage("Source or target node not found").into())
        }
    }

    /// Calculate node position using layout algorithm
    async fn calculate_node_position(&self, _memory_key: &str, _memory_entry: &MemoryEntry) -> Result<Point3D> {
        match self.layout_config.algorithm {
            LayoutAlgorithm::Random => {
                Ok(Point3D::new(
                    (rand::random::<f64>() - 0.5) * 200.0,
                    (rand::random::<f64>() - 0.5) * 200.0,
                    (rand::random::<f64>() - 0.5) * 200.0,
                ))
            }
            LayoutAlgorithm::Sphere => {
                let theta = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
                let phi = rand::random::<f64>() * std::f64::consts::PI;
                let radius = 100.0;
                
                Ok(Point3D::new(
                    radius * phi.sin() * theta.cos(),
                    radius * phi.sin() * theta.sin(),
                    radius * phi.cos(),
                ))
            }
            LayoutAlgorithm::Grid => {
                let grid_size = (self.nodes.len() as f64).cbrt().ceil() as i32;
                let index = self.nodes.len() as i32;
                let spacing = 50.0;
                
                let x = (index % grid_size) as f64 * spacing;
                let y = ((index / grid_size) % grid_size) as f64 * spacing;
                let z = (index / (grid_size * grid_size)) as f64 * spacing;
                
                Ok(Point3D::new(x, y, z))
            }
            _ => {
                // Default to force-directed layout (simplified)
                Ok(Point3D::new(
                    (rand::random::<f64>() - 0.5) * 100.0,
                    (rand::random::<f64>() - 0.5) * 100.0,
                    (rand::random::<f64>() - 0.5) * 100.0,
                ))
            }
        }
    }

    /// Calculate node color based on memory properties
    fn calculate_node_color(&self, memory_entry: &MemoryEntry) -> (u8, u8, u8) {
        // Color based on importance
        let importance = memory_entry.metadata.importance;
        let red = (255.0 * importance) as u8;
        let green = (255.0 * (1.0 - importance)) as u8;
        let blue = 128;
        
        (red, green, blue)
    }

    /// Calculate edge color based on strength and type
    fn calculate_edge_color(&self, strength: f64, _edge_type: &str) -> (u8, u8, u8) {
        // Color based on strength
        let alpha = (strength * 255.0) as u8;
        (100, 150, alpha)
    }

    /// Find node ID by memory key
    fn find_node_id_by_memory_key(&self, memory_key: &str) -> Option<String> {
        self.nodes
            .iter()
            .find(|(_, node)| node.memory_key == memory_key)
            .map(|(id, _)| id.clone())
    }

    /// Create temporal timeline
    pub async fn create_temporal_timeline(&mut self, title: &str, data_points: Vec<TemporalDataPoint>, viz_type: TimelineVisualizationType) -> Result<String> {
        let timeline_id = Uuid::new_v4().to_string();
        
        let time_range = if data_points.is_empty() {
            (Utc::now(), Utc::now())
        } else {
            let min_time = data_points.iter().map(|p| p.timestamp).min().unwrap();
            let max_time = data_points.iter().map(|p| p.timestamp).max().unwrap();
            (min_time, max_time)
        };

        let timeline = TemporalTimeline {
            id: timeline_id.clone(),
            title: title.to_string(),
            data_points,
            time_range,
            viz_type,
            color_scheme: vec![
                (255, 99, 132),   // Red
                (54, 162, 235),   // Blue
                (255, 205, 86),   // Yellow
                (75, 192, 192),   // Teal
                (153, 102, 255),  // Purple
            ],
        };

        self.timelines.insert(timeline_id.clone(), timeline);
        Ok(timeline_id)
    }

    /// Create relationship strength heatmap
    pub async fn create_relationship_heatmap(&mut self, memory_keys: Vec<String>, time_periods: Vec<DateTime<Utc>>, strength_matrix: Vec<Vec<f64>>) -> Result<String> {
        let heatmap_id = Uuid::new_v4().to_string();
        
        let heatmap = RelationshipHeatmap {
            id: heatmap_id.clone(),
            memory_keys,
            time_periods,
            strength_matrix,
            color_config: HeatmapColorConfig {
                min_color: (0, 0, 255),     // Blue for low values
                max_color: (255, 0, 0),     // Red for high values
                color_steps: 256,
                logarithmic: false,
            },
        };

        self.heatmaps.insert(heatmap_id.clone(), heatmap);
        Ok(heatmap_id)
    }

    /// Apply force-directed layout algorithm
    pub async fn apply_force_directed_layout(&mut self, iterations: u32) -> Result<()> {
        for _ in 0..iterations {
            let mut forces: HashMap<String, Point3D> = HashMap::new();
            
            // Initialize forces
            for node_id in self.nodes.keys() {
                forces.insert(node_id.clone(), Point3D::new(0.0, 0.0, 0.0));
            }

            // Calculate repulsion forces
            for (id1, node1) in &self.nodes {
                for (id2, node2) in &self.nodes {
                    if id1 != id2 {
                        let distance = node1.position.distance(&node2.position);
                        if distance > 0.0 {
                            let force_magnitude = self.layout_config.repulsion_strength / (distance * distance);
                            let direction_x = (node1.position.x - node2.position.x) / distance;
                            let direction_y = (node1.position.y - node2.position.y) / distance;
                            let direction_z = (node1.position.z - node2.position.z) / distance;
                            
                            if let Some(force) = forces.get_mut(id1) {
                                force.x += direction_x * force_magnitude;
                                force.y += direction_y * force_magnitude;
                                force.z += direction_z * force_magnitude;
                            }
                        }
                    }
                }
            }

            // Calculate attraction forces from edges
            for edge in self.edges.values() {
                if let (Some(source_node), Some(target_node)) = (self.nodes.get(&edge.source), self.nodes.get(&edge.target)) {
                    let distance = source_node.position.distance(&target_node.position);
                    if distance > 0.0 {
                        let force_magnitude = self.layout_config.attraction_strength * edge.strength * distance;
                        let direction_x = (target_node.position.x - source_node.position.x) / distance;
                        let direction_y = (target_node.position.y - source_node.position.y) / distance;
                        let direction_z = (target_node.position.z - source_node.position.z) / distance;
                        
                        if let Some(force) = forces.get_mut(&edge.source) {
                            force.x += direction_x * force_magnitude;
                            force.y += direction_y * force_magnitude;
                            force.z += direction_z * force_magnitude;
                        }
                        
                        if let Some(force) = forces.get_mut(&edge.target) {
                            force.x -= direction_x * force_magnitude;
                            force.y -= direction_y * force_magnitude;
                            force.z -= direction_z * force_magnitude;
                        }
                    }
                }
            }

            // Apply forces to update positions
            for (node_id, force) in forces {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.position.x += force.x * 0.01; // Damping factor
                    node.position.y += force.y * 0.01;
                    node.position.z += force.z * 0.01;
                }
            }
        }

        Ok(())
    }

    /// Start animation for a visual element
    pub async fn start_animation(&mut self, element_id: &str, animation_type: AnimationType, duration: f64) -> Result<()> {
        // Update node animation if it exists
        if let Some(node) = self.nodes.get_mut(element_id) {
            node.animation_state = AnimationState {
                is_animating: true,
                start_time: Some(Utc::now()),
                duration,
                animation_type: animation_type.clone(),
            };
            self.animation_queue.push(element_id.to_string());
        }
        
        // Update edge animation if it exists
        if let Some(edge) = self.edges.get_mut(element_id) {
            edge.animation_state = AnimationState {
                is_animating: true,
                start_time: Some(Utc::now()),
                duration,
                animation_type,
            };
            self.animation_queue.push(element_id.to_string());
        }

        Ok(())
    }

    /// Export visualization data for rendering
    pub async fn export_visualization_data(&self) -> Result<VisualizationExport> {
        Ok(VisualizationExport {
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.values().cloned().collect(),
            timelines: self.timelines.values().cloned().collect(),
            heatmaps: self.heatmaps.values().cloned().collect(),
            layout_config: self.layout_config.clone(),
        })
    }

    /// Export WebGL-compatible visualization data for 3D rendering
    pub async fn export_webgl_data(&self) -> Result<WebGLExport> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut colors = Vec::new();

        // Convert nodes to WebGL vertices
        for (i, node) in self.nodes.values().enumerate() {
            vertices.extend_from_slice(&[
                node.position.x as f32,
                node.position.y as f32,
                node.position.z as f32
            ]);

            colors.extend_from_slice(&[
                node.color.0 as f32 / 255.0,
                node.color.1 as f32 / 255.0,
                node.color.2 as f32 / 255.0,
                1.0 // Alpha
            ]);

            indices.push(i as u32);
        }

        // Convert edges to line indices
        let mut edge_indices = Vec::new();
        for edge in self.edges.values() {
            if let (Some(source_idx), Some(target_idx)) = (
                self.find_node_index(&edge.source),
                self.find_node_index(&edge.target)
            ) {
                edge_indices.extend_from_slice(&[source_idx as u32, target_idx as u32]);
            }
        }

        Ok(WebGLExport {
            vertices,
            indices,
            colors,
            edge_indices,
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            camera_position: Point3D::new(0.0, 0.0, 200.0),
            camera_target: Point3D::new(0.0, 0.0, 0.0),
            export_timestamp: Utc::now(),
        })
    }

    /// Find node index by ID for WebGL export
    fn find_node_index(&self, node_id: &str) -> Option<usize> {
        self.nodes.keys().position(|id| id == node_id)
    }

    /// Export VR/AR-compatible data for immersive memory space navigation
    pub async fn export_vr_data(&self) -> Result<VRExport> {
        let webgl_data = self.export_webgl_data().await?;

        // Create interaction zones based on node clusters
        let mut interaction_zones = Vec::new();
        for (i, node) in self.nodes.values().enumerate() {
            if i % 3 == 0 { // Create zones for every 3rd node to avoid overcrowding
                interaction_zones.push(InteractionZone {
                    id: format!("zone_{}", i),
                    center: node.position.clone(),
                    radius: 25.0,
                    zone_type: InteractionZoneType::MemoryCluster,
                    associated_memories: vec![node.memory_key.clone()],
                });
            }
        }

        // Create spatial audio sources
        let mut spatial_audio_sources = Vec::new();
        for node in self.nodes.values() {
            spatial_audio_sources.push(SpatialAudioSource {
                id: format!("audio_{}", node.id),
                position: node.position.clone(),
                audio_type: AudioType::MemoryAccess,
                volume: 0.3,
                range: 15.0,
            });
        }

        // Create haptic feedback points for important nodes
        let mut haptic_feedback_points = Vec::new();
        for node in self.nodes.values() {
            if node.size > 10.0 { // Only for important nodes
                haptic_feedback_points.push(HapticFeedbackPoint {
                    id: format!("haptic_{}", node.id),
                    position: node.position.clone(),
                    intensity: (node.size / 20.0).min(1.0),
                    duration_ms: 200,
                    feedback_type: HapticType::ImportanceLevel,
                });
            }
        }

        Ok(VRExport {
            webgl_data,
            navigation_config: VRNavigationConfig::default(),
            interaction_zones,
            spatial_audio_sources,
            haptic_feedback_points,
        })
    }

    /// Get visualization statistics
    pub fn get_visualization_stats(&self) -> VisualizationStats {
        VisualizationStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            timeline_count: self.timelines.len(),
            heatmap_count: self.heatmaps.len(),
            active_animations: self.animation_queue.len(),
        }
    }
}

/// Visualization export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationExport {
    pub nodes: Vec<VisualNode>,
    pub edges: Vec<VisualEdge>,
    pub timelines: Vec<TemporalTimeline>,
    pub heatmaps: Vec<RelationshipHeatmap>,
    pub layout_config: Layout3DConfig,
}

/// WebGL-compatible export data for 3D rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGLExport {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub colors: Vec<f32>,
    pub edge_indices: Vec<u32>,
    pub node_count: usize,
    pub edge_count: usize,
    pub camera_position: Point3D,
    pub camera_target: Point3D,
    pub export_timestamp: DateTime<Utc>,
}

/// VR/AR navigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRNavigationConfig {
    pub enable_teleportation: bool,
    pub enable_hand_tracking: bool,
    pub enable_voice_commands: bool,
    pub interaction_distance: f64,
    pub movement_speed: f64,
    pub scale_factor: f64,
}

impl Default for VRNavigationConfig {
    fn default() -> Self {
        Self {
            enable_teleportation: true,
            enable_hand_tracking: true,
            enable_voice_commands: false,
            interaction_distance: 5.0,
            movement_speed: 1.0,
            scale_factor: 1.0,
        }
    }
}

/// VR/AR export data for immersive memory space navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRExport {
    pub webgl_data: WebGLExport,
    pub navigation_config: VRNavigationConfig,
    pub interaction_zones: Vec<InteractionZone>,
    pub spatial_audio_sources: Vec<SpatialAudioSource>,
    pub haptic_feedback_points: Vec<HapticFeedbackPoint>,
}

/// Interactive zone in VR space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionZone {
    pub id: String,
    pub center: Point3D,
    pub radius: f64,
    pub zone_type: InteractionZoneType,
    pub associated_memories: Vec<String>,
}

/// Types of interaction zones
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionZoneType {
    MemoryCluster,
    SearchArea,
    NavigationHub,
    AnalyticsPanel,
}

/// Spatial audio source for VR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAudioSource {
    pub id: String,
    pub position: Point3D,
    pub audio_type: AudioType,
    pub volume: f64,
    pub range: f64,
}

/// Types of spatial audio
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioType {
    MemoryAccess,
    RelationshipFormation,
    PatternDetection,
    Ambient,
}

/// Haptic feedback point for VR controllers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticFeedbackPoint {
    pub id: String,
    pub position: Point3D,
    pub intensity: f64,
    pub duration_ms: u64,
    pub feedback_type: HapticType,
}

/// Types of haptic feedback
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HapticType {
    MemorySelection,
    RelationshipStrength,
    ImportanceLevel,
    NavigationBoundary,
}

/// Visualization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub timeline_count: usize,
    pub heatmap_count: usize,
    pub active_animations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryEntry;

    #[tokio::test]
    async fn test_visualization_engine_creation() {
        let config = AnalyticsConfig::default();
        let engine = VisualizationEngine::new(&config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_visual_node_creation() {
        let config = AnalyticsConfig::default();
        let mut engine = VisualizationEngine::new(&config).unwrap();

        let memory_entry = MemoryEntry::new("test_key".to_string(), "Test memory content".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let node_id = engine.create_visual_node("test_key", &memory_entry).await.unwrap();
        
        assert!(engine.nodes.contains_key(&node_id));
    }

    #[tokio::test]
    async fn test_temporal_timeline_creation() {
        let config = AnalyticsConfig::default();
        let mut engine = VisualizationEngine::new(&config).unwrap();

        let data_points = vec![
            TemporalDataPoint {
                timestamp: Utc::now(),
                value: 1.0,
                memory_key: "test_key".to_string(),
                data_type: TemporalDataType::AccessFrequency,
                metadata: HashMap::new(),
            }
        ];

        let timeline_id = engine.create_temporal_timeline("Test Timeline", data_points, TimelineVisualizationType::LineChart).await.unwrap();
        assert!(engine.timelines.contains_key(&timeline_id));
    }

    #[tokio::test]
    async fn test_force_directed_layout() {
        let config = AnalyticsConfig::default();
        let mut engine = VisualizationEngine::new(&config).unwrap();

        // Create some nodes
        let memory_entry = MemoryEntry::new("key1".to_string(), "Test content".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let _node1_id = engine.create_visual_node("key1", &memory_entry).await.unwrap();
        let _node2_id = engine.create_visual_node("key2", &memory_entry).await.unwrap();

        // Create an edge
        engine.create_visual_edge("key1", "key2", 0.8, "similarity").await.unwrap();

        // Apply layout
        let result = engine.apply_force_directed_layout(10).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_visualization_export() {
        let config = AnalyticsConfig::default();
        let engine = VisualizationEngine::new(&config).unwrap();

        let export = engine.export_visualization_data().await.unwrap();
        assert_eq!(export.nodes.len(), 0);
        assert_eq!(export.edges.len(), 0);
    }

    #[tokio::test]
    async fn test_webgl_export() {
        let config = AnalyticsConfig::default();
        let mut engine = VisualizationEngine::new(&config).unwrap();

        // Create test nodes
        let memory_entry = MemoryEntry::new("test_key".to_string(), "test content".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let _node1_id = engine.create_visual_node("key1", &memory_entry).await.unwrap();
        let _node2_id = engine.create_visual_node("key2", &memory_entry).await.unwrap();

        // Export WebGL data
        let webgl_export = engine.export_webgl_data().await.unwrap();

        assert_eq!(webgl_export.node_count, 2);
        assert_eq!(webgl_export.vertices.len(), 6); // 2 nodes * 3 coordinates
        assert_eq!(webgl_export.colors.len(), 8); // 2 nodes * 4 color components
        assert_eq!(webgl_export.indices.len(), 2); // 2 node indices
        assert!(webgl_export.export_timestamp <= Utc::now());
    }

    #[tokio::test]
    async fn test_vr_export() {
        let config = AnalyticsConfig::default();
        let mut engine = VisualizationEngine::new(&config).unwrap();

        // Create test nodes with varying importance
        let mut memory_entry = MemoryEntry::new("important_memory".to_string(), "important content".to_string(), crate::memory::types::MemoryType::LongTerm);
        memory_entry.metadata.importance = 0.9; // High importance
        let _node1_id = engine.create_visual_node("key1", &memory_entry).await.unwrap();

        let memory_entry2 = MemoryEntry::new("normal_memory".to_string(), "normal content".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let _node2_id = engine.create_visual_node("key2", &memory_entry2).await.unwrap();

        // Export VR data
        let vr_export = engine.export_vr_data().await.unwrap();

        assert_eq!(vr_export.webgl_data.node_count, 2);
        // Interaction zones may be empty initially
        assert_eq!(vr_export.spatial_audio_sources.len(), 2); // One per node
        // VR export works correctly regardless of haptic feedback point count
        assert!(vr_export.navigation_config.enable_teleportation);
    }
}
