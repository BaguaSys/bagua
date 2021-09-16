pub mod agent;

use crate::exporter::agent::{AgentAsyncClientHTTP, BaguaBatch, BaguaSpan};
use async_trait::async_trait;
use opentelemetry::{sdk::export::trace, Key};
use reqwest::StatusCode;
use std::time::UNIX_EPOCH;

#[derive(Debug)]
pub struct Exporter {
    pub uploader: AgentAsyncClientHTTP,
}

#[async_trait]
impl trace::SpanExporter for Exporter {
    async fn export(&mut self, batch: Vec<trace::SpanData>) -> trace::ExportResult {
        let mut bagua_spans = Vec::new();
        for span in batch {
            let bagua_span = BaguaSpan {
                trace_id: span.span_context.trace_id().to_u128(),
                action: span.name.into_owned(),
                tensor_name: span
                    .attributes
                    .get(&Key::new("tensor_name"))
                    .unwrap()
                    .as_str()
                    .to_string(),
                start_time: span
                    .start_time
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
                end_time: span
                    .end_time
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            };

            bagua_spans.push(bagua_span);
        }

        let resp = self
            .uploader
            .emit_batch(BaguaBatch { spans: bagua_spans })
            .await;
        match resp {
            Ok(resp) => {
                if resp.status() != StatusCode::OK {
                    tracing::warn!("upload bagua span failed, resp={:?}", resp);
                }
            }
            Err(err) => {
                tracing::warn!("upload bagua span failed, err={:?}", err);
            }
        }

        Ok(())
    }

    fn shutdown(&mut self) {}
}
