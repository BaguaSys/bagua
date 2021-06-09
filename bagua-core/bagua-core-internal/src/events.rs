use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BaguaEventChannel {
    inner: Arc<(Mutex<bool>, parking_lot::Condvar)>,
}

impl BaguaEventChannel {
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

impl Default for BaguaEventChannel {
    fn default() -> Self {
        Self {
            inner: Arc::new((Mutex::new(false), parking_lot::Condvar::new())),
        }
    }
}
