//! Box3D parallel_for.c: b3ParallelForShared, b3ParallelForTrampoline and b3ParallelFor.
//! Work must be independent of partition and execution order, which vary with
//! worker count. The scheduler must finish every trampoline handle before reuse.

use core::sync::atomic::{AtomicU32, Ordering};

/// Box3D blocksPerWorker: bounds claim overhead while allowing work stealing.
const BLOCKS_PER_WORKER: usize = 32;

/// Box3D physics_world.c's minimum per-contact collide task range.
pub const COLLIDE_MIN_RANGE: usize = 20;

pub struct ParFor {
    next_block: AtomicU32,
    block_count: usize,
    block_size: usize,
    item_count: usize,
}

impl ParFor {
    /// b3ParallelFor sizing: at most BLOCKS_PER_WORKER * worker_count blocks.
    /// The final block may contain fewer than min_range items.
    pub fn new(item_count: usize, min_range: usize, worker_count: usize) -> ParFor {
        debug_assert!(min_range > 0);
        debug_assert!(worker_count >= 1);

        let max_block_count = BLOCKS_PER_WORKER * worker_count;
        let (block_size, block_count) = if item_count == 0 {
            (min_range, 0)
        } else if item_count <= min_range * max_block_count {
            (min_range, item_count.div_ceil(min_range))
        } else {
            let size = item_count.div_ceil(max_block_count);
            (size, item_count.div_ceil(size))
        };

        ParFor {
            next_block: AtomicU32::new(0),
            block_count,
            block_size,
            item_count,
        }
    }

    /// Caps enqueued trampolines at the block count so no task is queued without work.
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Calls f with half-open ranges. All blocks have completed only after every
    /// concurrent run call returns; one caller's return is not a barrier.
    pub fn run(&self, f: impl Fn(usize, usize)) {
        loop {
            let index = self.next_block.fetch_add(1, Ordering::SeqCst) as usize;
            if index >= self.block_count {
                return;
            }
            let start = index * self.block_size;
            let end = (start + self.block_size).min(self.item_count);
            f(start, end);
        }
    }
}
