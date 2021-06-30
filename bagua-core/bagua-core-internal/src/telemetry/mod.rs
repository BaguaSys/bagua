use crate::BaguaCoreError;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use scheduled_thread_pool::ScheduledThreadPool;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
pub static SCHEDULED_THREAD_POOL: Lazy<ScheduledThreadPool> =
    Lazy::new(|| ScheduledThreadPool::with_name("bagua_scheduled_thread_pool", 1));

pub static TELEMETRY: Lazy<Option<Mutex<BaguaCommCoreTelemetry>>> =
    Lazy::new(|| match std::env::var("AUTO_TUNE_SERVER_ADDR") {
        Ok(x) => {
            tracing::info!("detected auto tuning server, connecting");
            Some(Mutex::new(BaguaCommCoreTelemetry::new(x.as_str())))
        }
        Err(_) => {
            tracing::warn!("auto tuning server not detected, may experience degraded performance");
            None
        }
    });

pub struct BaguaCommCoreTelemetry {
    server_addr: String,
    current_payload: TelemetryPayload,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TelemetryPayload {
    tensor_ready_order: Vec<String>,
    communication_time_ms: u64,
}

impl TelemetryPayload {
    pub fn clear(&mut self) {
        self.tensor_ready_order.clear();
        self.communication_time_ms = 0;
    }
}

impl Default for TelemetryPayload {
    fn default() -> Self {
        Self {
            tensor_ready_order: vec![],
            communication_time_ms: 0,
        }
    }
}

impl BaguaCommCoreTelemetry {
    pub fn new(server_addr: &str) -> Self {
        Self {
            server_addr: server_addr.to_string(),
            current_payload: TelemetryPayload::default(),
        }
    }

    pub fn new_tensor_ready(&mut self, tensor_name: &str) {
        self.current_payload
            .tensor_ready_order
            .push(tensor_name.to_string());
    }

    pub fn push_payload_and_clear(&mut self) -> Result<(), BaguaCoreError> {
        let payload_string = serde_json::to_string(&self.current_payload)?;
        if ureq::post(format!("{}/api/v1/bagua_backend_metrics", self.server_addr).as_str())
            .send_string(payload_string.as_str())?
            .status()
            != 200
        {
            return Err(BaguaCoreError::TelemetryError(
                "post TELEMETRY payload failed".into(),
            ));
        }
        self.clear();
        Ok(())
    }

    pub fn clear(&mut self) {
        self.current_payload.clear();
    }
}
