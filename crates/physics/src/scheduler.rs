//! Task-slot scheduler. The caller helps pending tasks in finish; only background workers park.
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

const MAX_TASKS: usize = 256;
const FREE: i32 = 0;
const PENDING: i32 = 1;
const CLAIMED: i32 = 2;
const COMPLETE: i32 = 3;
type Callback = unsafe fn(usize, usize);

struct Task {
    status: AtomicI32,
    data: UnsafeCell<Option<(Callback, usize, usize)>>,
}
impl Task {
    const fn new() -> Self {
        Self {
            status: AtomicI32::new(FREE),
            data: UnsafeCell::new(None),
        }
    }
}
// Only the enqueuing thread writes data, before publishing PENDING. The successful claimant reads it.
unsafe impl Sync for Task {}

pub struct Scheduler {
    tasks: [Task; MAX_TASKS],
    next_slot: AtomicUsize,
    semaphore: AtomicI32,
    shutdown: AtomicI32,
    fault: AtomicI32,
    live_workers: AtomicUsize,
}
impl Scheduler {
    pub const fn new() -> Self {
        Self {
            tasks: [const { Task::new() }; MAX_TASKS],
            next_slot: AtomicUsize::new(0),
            semaphore: AtomicI32::new(0),
            shutdown: AtomicI32::new(0),
            fault: AtomicI32::new(0),
            live_workers: AtomicUsize::new(0),
        }
    }
    pub fn task_count(&self) -> usize {
        self.next_slot.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        for task in &self.tasks[..self.next_slot.load(Ordering::SeqCst)] {
            assert_eq!(task.status.load(Ordering::SeqCst), COMPLETE);
            task.status.store(FREE, Ordering::SeqCst);
        }
        self.next_slot.store(0, Ordering::SeqCst);
    }
    pub fn start(&self) {
        self.semaphore.store(0, Ordering::SeqCst);
        self.shutdown.store(0, Ordering::SeqCst);
        self.fault.store(0, Ordering::SeqCst);
    }
    pub fn faulted(&self) -> bool {
        self.fault.load(Ordering::SeqCst) != 0
    }
    pub fn fault_ptr(&self) -> *mut i32 {
        self.fault.as_ptr()
    }
    pub fn register_worker(&self) {
        self.live_workers.fetch_add(1, Ordering::SeqCst);
    }
    pub fn fail(&self) {
        self.fault.store(1, Ordering::SeqCst);
        self.stop();
        self.live_workers.fetch_sub(1, Ordering::SeqCst);
    }
    pub fn join_fault(&self) {
        let mut spins = 1;
        while self.live_workers.load(Ordering::SeqCst) != 0 {
            pause(spins);
            spins = (spins * 2).min(64);
        }
    }
    pub fn stop(&self) {
        self.shutdown.store(1, Ordering::SeqCst);
        for _ in 0..8 {
            self.post();
        }
    }
    fn post(&self) {
        self.semaphore.fetch_add(1, Ordering::SeqCst);
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        unsafe {
            core::arch::wasm32::memory_atomic_notify(self.semaphore.as_ptr(), 1);
        }
    }
    fn wait(&self) {
        loop {
            let value = self.semaphore.load(Ordering::SeqCst);
            if value > 0
                && self
                    .semaphore
                    .compare_exchange(value, value - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                return;
            }
            if value == 0 {
                #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
                unsafe {
                    core::arch::wasm32::memory_atomic_wait32(self.semaphore.as_ptr(), 0, -1);
                }
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::yield_now();
            }
        }
    }
    /// The context must outlive finish. Only the stepping thread enqueues or resets slots.
    pub unsafe fn enqueue(
        &self,
        callback: Callback,
        context: usize,
        index: usize,
    ) -> Option<usize> {
        let slot = self.next_slot.load(Ordering::SeqCst);
        if slot == MAX_TASKS {
            callback(context, index);
            return None;
        }
        let task = &self.tasks[slot];
        *task.data.get() = Some((callback, context, index));
        self.next_slot.store(slot + 1, Ordering::SeqCst);
        task.status.store(PENDING, Ordering::SeqCst);
        self.post();
        Some(slot)
    }
    fn execute_one(&self) -> bool {
        if self.faulted() {
            return false;
        }
        for task in &self.tasks[..self.next_slot.load(Ordering::SeqCst)] {
            if task
                .status
                .compare_exchange(PENDING, CLAIMED, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                continue;
            }
            let (callback, context, index) = unsafe { (*task.data.get()).unwrap() };
            unsafe {
                callback(context, index);
            }
            task.status.store(COMPLETE, Ordering::SeqCst);
            return true;
        }
        false
    }
    pub fn finish(&self, handle: Option<usize>) {
        let Some(slot) = handle else {
            return;
        };
        let mut spins = 1;
        while self.tasks[slot].status.load(Ordering::SeqCst) != COMPLETE {
            if self.faulted() {
                self.join_fault();
                return;
            }
            if self.execute_one() {
                spins = 1;
            } else {
                pause(spins);
                spins = (spins * 2).min(64);
            }
        }
    }

    pub fn worker_with_fault(&self, inject_fault: bool) {
        loop {
            self.wait();
            if self.shutdown.load(Ordering::SeqCst) != 0 {
                self.live_workers.fetch_sub(1, Ordering::SeqCst);
                return;
            }
            if inject_fault {
                #[cfg(target_arch = "wasm32")]
                core::arch::wasm32::unreachable();
                #[cfg(not(target_arch = "wasm32"))]
                panic!("injected worker trap");
            }
            while self.execute_one() {}
        }
    }
}
pub fn pause(spins: u32) {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    unsafe {
        #[link(wasm_import_module = "env")]
        extern "C" {
            fn solverPause();
        }
        for _ in 0..spins {
            solverPause();
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    for _ in 0..spins {
        core::hint::spin_loop();
    }
}
