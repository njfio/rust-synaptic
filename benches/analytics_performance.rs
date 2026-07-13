// Benchmark code: unwrap on setup failure is acceptable.
#![allow(clippy::unwrap_used, clippy::panic)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use synaptic::analytics::{
    AccessType, AnalyticsConfig, AnalyticsEngine, AnalyticsEvent, ModificationType,
};
use tokio::runtime::Runtime;

/// Build a fresh analytics engine for benchmarking.
fn new_engine() -> AnalyticsEngine {
    AnalyticsEngine::new(AnalyticsConfig::default()).expect("analytics engine should initialize")
}

/// Create a synthetic memory-access event for a given index.
fn access_event(i: usize) -> AnalyticsEvent {
    let access_type = match i % 6 {
        0 => AccessType::Read,
        1 => AccessType::Write,
        2 => AccessType::Update,
        3 => AccessType::Delete,
        4 => AccessType::Search,
        _ => AccessType::Traverse,
    };

    AnalyticsEvent::MemoryAccess {
        memory_key: format!("analytics_entry_{}", i),
        access_type,
        timestamp: chrono::Utc::now(),
        user_context: Some(format!("user_{}", i % 10)),
    }
}

/// Benchmark analytics calculations (event ingestion + insight generation).
fn bench_analytics_calculations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("analytics_calculations");

    for data_size in [100, 500, 1000, 2000].iter() {
        let data_size = *data_size;
        group.throughput(Throughput::Elements(data_size as u64));

        // Benchmark memory pattern analysis via event ingestion + insights.
        group.bench_with_input(
            BenchmarkId::new("memory_pattern_analysis", data_size),
            &data_size,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut engine = new_engine();

                        for i in 0..count {
                            engine
                                .record_event(black_box(access_event(i)))
                                .await
                                .unwrap();
                        }

                        engine.generate_insights().await.unwrap();
                    })
                })
            },
        );

        // Benchmark trend analysis via modification events spread over time.
        group.bench_with_input(
            BenchmarkId::new("trend_analysis", data_size),
            &data_size,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut engine = new_engine();
                        let base = chrono::Utc::now();

                        for i in 0..count {
                            let event = AnalyticsEvent::MemoryModification {
                                memory_key: format!("analytics_entry_{}", i),
                                modification_type: ModificationType::ContentUpdate,
                                timestamp: base + chrono::Duration::seconds(i as i64),
                                change_magnitude: (i % 100) as f64 / 100.0,
                            };
                            engine.record_event(black_box(event)).await.unwrap();
                        }

                        engine.generate_insights().await.unwrap();
                    })
                })
            },
        );

        // Benchmark clustering analysis via search events + insight generation.
        group.bench_with_input(
            BenchmarkId::new("clustering_analysis", data_size),
            &data_size,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut engine = new_engine();

                        for i in 0..count {
                            let event = AnalyticsEvent::SearchQuery {
                                query: format!("cluster query {}", i % 5),
                                results_count: i % 20,
                                timestamp: chrono::Utc::now(),
                                response_time_ms: (i % 50) as u64,
                            };
                            engine.record_event(black_box(event)).await.unwrap();
                        }

                        black_box(engine.generate_insights().await.unwrap());
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark real-time analytics.
fn bench_realtime_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("realtime_analytics");

    // Benchmark incremental analytics updates: record an event and generate
    // insights after each new arrival.
    group.bench_function("incremental_analytics_update", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();

                // Initial data
                for i in 0..100 {
                    engine.record_event(access_event(i)).await.unwrap();
                }

                // Simulate real-time updates with incremental insight generation.
                for i in 100..150 {
                    engine
                        .record_event(black_box(access_event(i)))
                        .await
                        .unwrap();
                    engine.generate_insights().await.unwrap();
                }
            })
        })
    });

    // Benchmark streaming analytics: process events in batches.
    group.bench_function("streaming_analytics", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();

                for batch in 0..10 {
                    for i in 0..20 {
                        let event = AnalyticsEvent::SearchQuery {
                            query: format!("streaming batch {} item {}", batch, i),
                            results_count: i,
                            timestamp: chrono::Utc::now(),
                            response_time_ms: (i % 30) as u64,
                        };
                        engine.record_event(black_box(event)).await.unwrap();
                    }

                    // Process batch analytics.
                    engine.generate_insights().await.unwrap();
                }
            })
        })
    });

    group.finish();
}

/// Benchmark memory optimization analytics.
fn bench_optimization_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("optimization_analytics");

    // Benchmark memory usage analysis via varied modification magnitudes.
    group.bench_function("memory_usage_analysis", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();

                for i in 0..500 {
                    let size_factor = (i % 10) + 1;
                    let event = AnalyticsEvent::MemoryModification {
                        memory_key: format!("usage_entry_{}", i),
                        modification_type: ModificationType::ContentUpdate,
                        timestamp: chrono::Utc::now(),
                        change_magnitude: (size_factor * 100) as f64,
                    };
                    engine.record_event(black_box(event)).await.unwrap();
                }

                engine.generate_insights().await.unwrap();
            })
        })
    });

    // Benchmark performance bottleneck detection via search response times.
    group.bench_function("bottleneck_detection", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();

                for i in 0..200 {
                    let event = AnalyticsEvent::SearchQuery {
                        query: format!("perf data point {}", i),
                        results_count: i % 100,
                        timestamp: chrono::Utc::now(),
                        response_time_ms: (i % 100) as u64,
                    };
                    engine.record_event(black_box(event)).await.unwrap();
                }

                black_box(engine.generate_insights().await.unwrap());
            })
        })
    });

    group.finish();
}

/// Benchmark predictive analytics.
fn bench_predictive_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("predictive_analytics");

    // Benchmark memory growth prediction over a growing access history.
    group.bench_function("memory_growth_prediction", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();
                let base = chrono::Utc::now();

                for day in 0..30 {
                    let daily_entries = 50 + (day * 2); // Growing pattern
                    for entry in 0..daily_entries {
                        let event = AnalyticsEvent::MemoryAccess {
                            memory_key: format!("day_{}_entry_{}", day, entry),
                            access_type: AccessType::Write,
                            timestamp: base + chrono::Duration::days(day as i64),
                            user_context: None,
                        };
                        engine.record_event(black_box(event)).await.unwrap();
                    }
                }

                // Generate insights (includes predictive growth analysis).
                engine.generate_insights().await.unwrap();
            })
        })
    });

    // Benchmark access pattern prediction.
    group.bench_function("access_pattern_prediction", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();

                for i in 0..300 {
                    let event = AnalyticsEvent::MemoryAccess {
                        memory_key: format!("access_pattern_{}", i % 20),
                        access_type: AccessType::Read,
                        timestamp: chrono::Utc::now(),
                        user_context: Some(format!("ctx_{}", i % 5)),
                    };
                    engine.record_event(black_box(event)).await.unwrap();
                }

                black_box(engine.generate_insights().await.unwrap());
            })
        })
    });

    group.finish();
}

/// Benchmark analytics aggregation.
fn bench_analytics_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("analytics_aggregation");

    // Benchmark time-series aggregation over a day of minute-resolution events.
    group.bench_function("timeseries_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = new_engine();
                let base = chrono::Utc::now();

                for hour in 0..24 {
                    for minute in 0..60 {
                        let offset = chrono::Duration::minutes((hour * 60 + minute) as i64);
                        let event = AnalyticsEvent::MemoryAccess {
                            memory_key: format!("ts_{}_{:02}", hour, minute),
                            access_type: AccessType::Read,
                            timestamp: base + offset,
                            user_context: None,
                        };
                        engine.record_event(black_box(event)).await.unwrap();
                    }
                }

                // Aggregate via insight generation.
                black_box(engine.generate_insights().await.unwrap());
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_analytics_calculations,
    bench_realtime_analytics,
    bench_optimization_analytics,
    bench_predictive_analytics,
    bench_analytics_aggregation
);
criterion_main!(benches);
