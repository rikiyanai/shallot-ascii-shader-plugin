//! Box3D table.c: symmetric pair keys, linear probing and backward-shift deletion.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Item {
    pub key: u64,
    pub hash: u32,
    padding: u32,
}

fn item_slot(items: &[Item], key: u64, hash: u32) -> usize {
    let mask = items.len() - 1;
    let mut index = hash as usize & mask;
    while items[index].hash != 0 && items[index].key != key {
        index = (index + 1) & mask;
    }
    index
}
fn item_key(a: u32, b: u32, child: u32) -> u64 {
    ((pair_key_hi(a, b) as u64) << 32) | pair_key_lo(a, b, child) as u64
}
fn item_insert(items: &mut [Item], key: u64, hash: u32) {
    let index = item_slot(items, key, hash);
    items[index].key = key;
    items[index].hash = hash;
}
pub fn transfer_items(old: &[Item], new: &mut [Item]) {
    for item in old {
        if item.hash != 0 {
            item_insert(new, item.key, item.hash);
        }
    }
}
pub fn contains_item(items: &[Item], a: u32, b: u32, child: u32) -> bool {
    if items.is_empty() {
        return false;
    }
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    items[item_slot(items, key, hash)].key == key
}

const SHAPE_MASK: u32 = (1 << 22) - 1;
const CHILD_MASK: u32 = (1 << 20) - 1;

pub fn pair_key_hi(a: u32, b: u32) -> u32 {
    ((a.min(b) & SHAPE_MASK) << 10) | ((a.max(b) & SHAPE_MASK) >> 12)
}
pub fn pair_key_lo(a: u32, b: u32, child: u32) -> u32 {
    ((a.max(b) & 0xfff) << 20) | (child & CHILD_MASK)
}
pub fn key_hash(hi: u32, lo: u32) -> u32 {
    let mut h = ((hi as u64) << 32) | lo as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h as u32
}


fn capacity(need: usize) -> usize {
    need.max(16).next_power_of_two()
}

#[cfg(target_arch = "wasm32")]
pub extern "C" fn create_set_in_world(world_index: usize, need: usize) {
    crate::broad::reserve_broad_in_world(world_index, 0, 0, 0, capacity(need));
}
#[cfg(target_arch = "wasm32")]
#[export_name = "broadEnsureSet"]
pub extern "C" fn ensure_set(need: usize) {
    ensure_set_in_world(crate::regions::active(), need)
}
#[cfg(target_arch = "wasm32")]
pub extern "C" fn ensure_set_in_world(world_index: usize, need: usize) {
    if crate::broad::set_cap(world_index) == 0 {
        create_set_in_world(world_index, need);
    }
}
#[cfg(target_arch = "wasm32")]
unsafe fn resident(world_index: usize) -> &'static mut [Item] {
    core::slice::from_raw_parts_mut(
        crate::broad::set_items(world_index),
        crate::broad::set_cap(world_index),
    )
}
/// # Safety
/// The active world must be selected, and this must run at a serial point, since growth moves the set.
#[cfg(target_arch = "wasm32")]
#[export_name = "broadAddPair"]
pub unsafe extern "C" fn add_pair(a: u32, b: u32, child: u32) -> u32 {
    unsafe { add_pair_in_world(crate::regions::active(), a, b, child) }
}
#[cfg(target_arch = "wasm32")]
/// # Safety
/// `world_index` must identify a live world, exclusively accessed while its set may grow.
pub unsafe extern "C" fn add_pair_in_world(world_index: usize, a: u32, b: u32, child: u32) -> u32 {
    ensure_set_in_world(world_index, 16);
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    let items = resident(world_index);
    if items[item_slot(items, key, hash)].hash != 0 {
        return 1;
    }
    if 2 * crate::broad::set_count(world_index) >= items.len() {
        crate::broad::grow_set(world_index);
    }
    item_insert(resident(world_index), key, hash);
    crate::broad::change_set_count(world_index, 1);
    0
}
/// # Safety
/// The active world must be selected, and no other thread may touch the set while this runs.
#[cfg(target_arch = "wasm32")]
#[export_name = "broadRemovePair"]
pub unsafe extern "C" fn remove_pair(a: u32, b: u32, child: u32) -> u32 {
    unsafe { remove_pair_in_world(crate::regions::active(), a, b, child) }
}
#[cfg(target_arch = "wasm32")]
/// # Safety
/// `world_index` must identify a live world, with no concurrent access to its set.
pub unsafe extern "C" fn remove_pair_in_world(
    world_index: usize,
    a: u32,
    b: u32,
    child: u32,
) -> u32 {
    if crate::broad::set_cap(world_index) == 0 {
        return 0;
    }
    let items = resident(world_index);
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    let mut i = item_slot(items, key, hash);
    if items[i].hash == 0 {
        return 0;
    }
    items[i].key = 0;
    items[i].hash = 0;
    crate::broad::change_set_count(world_index, -1);
    let mask = items.len() - 1;
    let mut j = i;
    loop {
        j = (j + 1) & mask;
        if items[j].hash == 0 {
            break;
        }
        let k = items[j].hash as usize & mask;
        if if i <= j {
            i < k && k <= j
        } else {
            i < k || k <= j
        } {
            continue;
        }
        items[i] = items[j];
        items[j].key = 0;
        items[j].hash = 0;
        i = j;
    }
    1
}
