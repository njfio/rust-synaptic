#[cfg(feature = "multimodal")]
use synaptic::multimodal::{data::DataMemoryProcessor, ContentType, MultiModalProcessor};

#[cfg(feature = "multimodal")]
#[tokio::test]
async fn test_data_processor_processing_time() {
    let processor = DataMemoryProcessor::default();
    let csv = b"a,b\n1,2\n3,4";
    let content_type = ContentType::Data { format: "CSV".to_string(), schema: None };
    let memory = processor.process(csv, &content_type).await.unwrap();
    assert!(memory.metadata.processing_time_ms > 0);
    assert!(memory.metadata.processing_time_ms < 10_000);
}

