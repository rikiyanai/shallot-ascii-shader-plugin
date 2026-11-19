use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, Layout};

unsafe fn allocate(size: usize) -> *mut u8 {
    if size == 0 {
        return 16 as *mut u8;
    }
    let layout = Layout::from_size_align(size, 16).unwrap();
    let data = alloc(layout);
    if data.is_null() {
        handle_alloc_error(layout);
    }
    data
}

unsafe fn release(data: *mut u8, size: usize) {
    if size != 0 {
        dealloc(data, Layout::from_size_align(size, 16).unwrap());
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StackEntry {
    data: *mut u8,
    name: *const u8,
    size: usize,
    used_malloc: bool,
}

#[repr(C)]
pub(crate) struct Stack {
    memory: *mut u8,
    capacity: usize,
    index: usize,
    allocation: usize,
    max_allocation: usize,
    entries: [StackEntry; 32],
    entry_count: usize,
}

impl Stack {
    pub(crate) const EMPTY: Self = Self {
        memory: 16 as *mut u8,
        capacity: 0,
        index: 0,
        entries: [StackEntry {
            data: 16 as *mut u8,
            name: std::ptr::null(),
            size: 0,
            used_malloc: false,
        }; 32],
        entry_count: 0,
        allocation: 0,
        max_allocation: 0,
    };

    pub(crate) fn max_allocation(&self) -> usize {
        self.max_allocation
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    pub(crate) unsafe fn new(capacity: usize) -> Self {
        Self {
            memory: allocate(capacity),
            capacity,
            ..Self::EMPTY
        }
    }

    pub(crate) unsafe fn alloc(&mut self, size: usize) -> *mut u8 {
        assert!(self.entry_count < self.entries.len());
        let size = size.next_multiple_of(16);
        let used_malloc = self.index + size > self.capacity;
        let data = if used_malloc {
            allocate(size)
        } else {
            let data = self.memory.add(self.index);
            self.index += size;
            data
        };
        self.allocation += size;
        self.max_allocation = self.max_allocation.max(self.allocation);
        self.entries[self.entry_count] = StackEntry {
            data,
            name: c"step columns".as_ptr().cast(),
            size,
            used_malloc,
        };
        self.entry_count += 1;
        data
    }

    pub(crate) unsafe fn free(&mut self, data: *mut u8) {
        assert!(self.entry_count > 0);
        let entry = self.entries[self.entry_count - 1];
        assert_eq!(data, entry.data);
        if entry.used_malloc {
            release(data, entry.size);
        } else {
            self.index -= entry.size;
        }
        self.allocation -= entry.size;
        self.entry_count -= 1;
    }

    pub(crate) unsafe fn grow(&mut self) {
        assert_eq!(self.allocation, 0);
        if self.max_allocation > self.capacity {
            release(self.memory, self.capacity);
            self.capacity = self.max_allocation + self.max_allocation / 2;
            self.memory = allocate(self.capacity);
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        assert_eq!(self.entry_count, 0);
        unsafe { release(self.memory, self.capacity) };
    }
}

unsafe fn allocate_arena_memory(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size, 16).unwrap();
    let data = alloc_zeroed(layout);
    if data.is_null() {
        handle_alloc_error(layout);
    }
    data
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OverflowBlock {
    data: *mut u8,
    size: usize,
}

#[repr(C)]
struct OverflowArray {
    data: *mut OverflowBlock,
    count: usize,
    capacity: usize,
}

impl OverflowArray {
    unsafe fn push(&mut self, block: OverflowBlock) {
        if self.count == self.capacity {
            let capacity = if self.capacity == 0 {
                8
            } else {
                2 * self.capacity
            };
            let data =
                allocate(capacity * std::mem::size_of::<OverflowBlock>()).cast::<OverflowBlock>();
            if self.count > 0 {
                std::ptr::copy_nonoverlapping(self.data, data, self.count);
                release(
                    self.data.cast(),
                    self.capacity * std::mem::size_of::<OverflowBlock>(),
                );
            }
            self.data = data;
            self.capacity = capacity;
        }
        self.data.add(self.count).write(block);
        self.count += 1;
    }


}

#[repr(C)]
struct ArenaSharedState {
    overflows: OverflowArray,
    max_index: usize,
    overflow_bytes: usize,
    peak_demand: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Arena {
    memory: *mut u8,
    capacity: usize,
    index: usize,
    shared: *mut ArenaSharedState,
}

pub(crate) struct WorkerArena {
    pub(crate) arena: Arena,
}

impl WorkerArena {
    pub(crate) unsafe fn new(capacity: usize) -> Self {
        let capacity = capacity.max(8);
        let shared = allocate(std::mem::size_of::<ArenaSharedState>()).cast::<ArenaSharedState>();
        shared.write(ArenaSharedState {
            overflows: OverflowArray {
                data: std::ptr::null_mut(),
                count: 0,
                capacity: 0,
            },
            max_index: 0,
            overflow_bytes: 0,
            peak_demand: 0,
        });
        Self {
            arena: Arena {
                memory: allocate_arena_memory(capacity),
                capacity,
                index: 0,
                shared,
            },
        }
    }

    pub(crate) unsafe fn peak_demand(&self) -> usize {
        (*self.arena.shared).peak_demand
    }

    pub(crate) unsafe fn sync(&mut self) {
        let arena = &mut self.arena;
        let shared = &mut *arena.shared;
        for i in 0..shared.overflows.count {
            let block = *shared.overflows.data.add(i);
            release(block.data, block.size);
        }
        shared.overflows.count = 0;
        let demand = shared.max_index + shared.overflow_bytes;
        shared.peak_demand = shared.peak_demand.max(demand);
        if demand > arena.capacity {
            release(arena.memory, arena.capacity);
            arena.capacity = demand + demand / 2;
            arena.memory = allocate_arena_memory(arena.capacity);
        }
        arena.index = 0;
        shared.max_index = 0;
        shared.overflow_bytes = 0;
    }
}

impl Drop for WorkerArena {
    fn drop(&mut self) {
        unsafe {
            let shared = &mut *self.arena.shared;
            for i in 0..shared.overflows.count {
                let block = *shared.overflows.data.add(i);
                release(block.data, block.size);
            }
            release(
                shared.overflows.data.cast(),
                shared.overflows.capacity * std::mem::size_of::<OverflowBlock>(),
            );
            release(
                self.arena.shared.cast(),
                std::mem::size_of::<ArenaSharedState>(),
            );
            release(self.arena.memory, self.arena.capacity);
        }
    }
}

impl Arena {
    pub(crate) unsafe fn bump(&mut self, size: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::null_mut();
        }
        let aligned = self.index.next_multiple_of(16);
        let shared = &mut *self.shared;
        if aligned + size > self.capacity {
            let data = allocate_arena_memory(size);
            shared.overflows.push(OverflowBlock { data, size });
            shared.overflow_bytes += size;
            return data;
        }
        self.index = aligned + size;
        shared.max_index = shared.max_index.max(self.index);
        self.memory.add(aligned)
    }

    // T must admit every initialized byte pattern. Arena backing is initialized once on
    // creation/growth so typed borrowed views never cover uninitialized storage.
    pub(crate) unsafe fn span<'a, T>(&mut self, count: usize) -> &'a mut [T] {
        assert!(std::mem::align_of::<T>() <= 16);
        if count == 0 {
            return &mut [];
        }
        let data = self.bump(count * std::mem::size_of::<T>()).cast::<T>();
        std::slice::from_raw_parts_mut(data, count)
    }
}
