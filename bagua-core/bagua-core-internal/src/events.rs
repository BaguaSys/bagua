use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BaguaEventChannel {
    pub name: String,
    inner: Arc<(Mutex<bool>, parking_lot::Condvar)>,
}

impl BaguaEventChannel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            inner: Arc::new((Mutex::new(false), parking_lot::Condvar::new())),
        }
    }

    pub fn finish(&self) {
        let &(ref lock, ref cvar) = &*self.inner;
        let mut finished = lock.lock();
        *finished = true;
        cvar.notify_all();
    }

    pub fn wait(&self) {
        let &(ref lock, ref cvar) = &*self.inner;
        let mut finished = lock.lock();
        if !*finished {
            cvar.wait(&mut finished);
        }
    }
}
