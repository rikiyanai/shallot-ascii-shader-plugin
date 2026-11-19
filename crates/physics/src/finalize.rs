//! Box3D solver.c b3FinalizeBodiesTask.
//! Preserve scalar operand order and intermediate rounding without FMA contraction.
//! Shape bounds commit follows finalize; proxy enlargement runs serially.

#[cfg(target_arch = "wasm32")]
use crate::body::{
    clear_sim_force_torque, flags::DYNAMIC, read_fin, read_sim, read_state, write_sim_center,
    write_sim_inv_inertia_world, write_sim_rotation, write_sim_transform_p, BODY_ID, CENTER0,
    FLAGS, MIN_EXTENT, ROTATION0, SIM_STRIDE,
};
#[cfg(target_arch = "wasm32")]
use crate::col::Col;
use crate::math::{maxf, minf, Mat3, Transform, Vec3};

/// Position-correction weight for the sleep-velocity blend (b3FinalizeBodies `positionSleepFactor`).
const POSITION_SLEEP_FACTOR: f32 = 0.5;

// The arena commits bounds and enlarge flags to the owning shape columns.

/// Shape type codes (b3ShapeType, types.h:434-453), as stored at `shapes::S_TYPE` and in a compound
/// child's first word.
pub const TY_CAPSULE: u32 = 0;
pub const TY_COMPOUND: u32 = 1;
pub const TY_HEIGHT_FIELD: u32 = 2;
pub const TY_HULL: u32 = 3;
pub const TY_MESH: u32 = 4;
pub const TY_SPHERE: u32 = 5;

/// Speculative AABB margin (B3_SPECULATIVE_DISTANCE).
const SPECULATIVE_DISTANCE: f32 = 4.0 * 0.005;

#[inline]
pub fn is_convex_refit(shape_type: u32) -> bool {
    matches!(shape_type, TY_SPHERE | TY_CAPSULE | TY_HULL)
}

/// b3AABB_Contains. Bounds are [lower.xyz, upper.xyz]; containment is inclusive.
#[inline]
pub fn aabb_contains(a: &[f32; 6], b: &[f32; 6]) -> bool {
    !(a[0] > b[0] || b[3] > a[3] || a[1] > b[1] || b[4] > a[4] || a[2] > b[2] || b[5] > a[5])
}

/// Tight world AABB of a sphere under `xf` (b3ComputeSphereAABB / `computeSphereAABBOut`). `geom` is
/// the shape column payload: center(3) radius(1).
#[inline]
fn sphere_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let c = xf.q.rotate(Vec3::new(geom[0], geom[1], geom[2])).add(xf.p);
    let r = geom[3];
    (
        Vec3::new(c.x - r, c.y - r, c.z - r),
        Vec3::new(c.x + r, c.y + r, c.z + r),
    )
}

/// Tight world AABB of a capsule under `xf` (b3ComputeCapsuleAABB / `computeCapsuleAABBOut`). `geom` is
/// center1(3) center2(3) radius(1); the min/max are the `b3Min`/`b3Max` ternaries (`minf`/`maxf`).
#[inline]
fn capsule_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let c1 = xf.q.rotate(Vec3::new(geom[0], geom[1], geom[2])).add(xf.p);
    let c2 = xf.q.rotate(Vec3::new(geom[3], geom[4], geom[5])).add(xf.p);
    let r = geom[6];
    (
        Vec3::new(
            minf(c1.x, c2.x) - r,
            minf(c1.y, c2.y) - r,
            minf(c1.z, c2.z) - r,
        ),
        Vec3::new(
            maxf(c1.x, c2.x) + r,
            maxf(c1.y, c2.y) + r,
            maxf(c1.z, c2.z) + r,
        ),
    )
}

/// Tight world AABB of a hull under `xf` (b3AABB_Transform of the hull's local AABB / `transformOut`).
/// `geom` is the local AABB: lower(3) upper(3) — the only hull field the AABB path reads.
#[inline]
fn hull_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let lo = Vec3::new(geom[0], geom[1], geom[2]);
    let hi = Vec3::new(geom[3], geom[4], geom[5]);
    let center = xf.q.rotate(hi.add(lo).scale(0.5)).add(xf.p);
    let extent = Mat3::from_quat(xf.q).abs().mul_v(hi.sub(lo).scale(0.5));
    (center.sub(extent), center.add(extent))
}

/// Inflate a tight shape bound by the speculative distance and test its resident fat margin.
pub fn refit_bounds(b: [f32; 6], fat: &[f32; 6]) -> ([f32; 6], bool) {
    let lo = Vec3::new(b[0], b[1], b[2]);
    let hi = Vec3::new(b[3], b[4], b[5]);
    let s = SPECULATIVE_DISTANCE;
    let cand = [lo.x - s, lo.y - s, lo.z - s, hi.x + s, hi.y + s, hi.z + s];
    let escaped = !aabb_contains(fat, &cand);
    (cand, escaped)
}

pub fn convex_bounds(shape_type: u32, geom: &[f32], xf: Transform) -> [f32; 6] {
    let (lo, hi) = match shape_type {
        TY_SPHERE => sphere_aabb(geom, xf),
        TY_CAPSULE => capsule_aabb(geom, xf),
        _ => hull_aabb(geom, xf),
    };
    [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
}

/// Advance the bodies in `[start, start+count)` from their solved velocity/position deltas.
///
/// Reads state (velocities + deltas) and sim (transform.q, invInertiaLocal, center, localCenter,
/// maxExtent, minExtent and flags). Writes the advanced center, transform.p, transform.q and
/// invInertiaWorld, the cleared force/torque (sim), the reset deltas (state), sleepVelocity on the
/// owning body, and the sweep base for
/// non-fast bodies. Fast non-bullets sweep immediately; bullets retain their base for the deferred
/// sweep. `h` is the full-step dt, `inv_dt` its inverse.
///
/// # Safety
/// Resident body records and continuous scratch must be reserved. Only this worker may write
/// bodies in the requested range, and no worker may grow memory while it runs.
#[cfg(target_arch = "wasm32")]
pub unsafe fn finalize(
    world_index: usize,
    worker: usize,
    state_col: Col<f32>,
    sim_col: Col<f32>,
    flags_col: Col<u32>,
    start: usize,
    count: usize,
    h: f32,
    inv_dt: f32,
    enable_continuous: bool,
) {
    for i in start..start + count {
        let s = read_state(state_col, i);
        let sim = read_sim(sim_col, i);
        let fin = read_fin(sim_col, i);

        let v = s.linear_velocity;
        let w = s.angular_velocity;
        let q0 = sim.rotation;

        // Velocity of the farthest point accounts for rotation; both arcs are measured in the
        // pre-advance frame, so they read the old rotation.
        let local_omega = q0.inv_rotate(w);
        let local_delta_rotation = q0.inv_rotate(s.delta_rotation.v);

        let center = fin.center.add(s.delta_position); // b3OffsetPos
        let q = s.delta_rotation.mul(q0).normalize();

        let velocity_arc = local_omega.abs().modified_cross(fin.max_extent);
        let max_velocity = v.length() + velocity_arc.length();

        // For small angles |theta| ~= 2 * length(sin(theta/2) * v), hence the 2x on the rotation arc.
        let rotation_arc = local_delta_rotation.abs().modified_cross(fin.max_extent);
        let max_delta_position = s.delta_position.length() + 2.0 * rotation_arc.length();

        // Position correction matters less than true velocity for sleep.
        let sleep_velocity = maxf(
            max_velocity,
            POSITION_SLEEP_FACTOR * inv_dt * max_delta_position,
        );

        let state_offset = i * crate::body::STATE_STRIDE;
        for n in 6..12 {
            state_col.set(state_offset + n, 0.0);
        }
        state_col.set(state_offset + 12, 1.0);

        let transform_p = center.add(q.rotate(fin.local_center).neg());

        write_sim_rotation(sim_col, i, q);
        clear_sim_force_torque(sim_col, i);
        write_sim_center(sim_col, i, center);
        write_sim_transform_p(sim_col, i, transform_p);

        let o = i * SIM_STRIDE;
        let body_id = sim_col.get(o + BODY_ID).to_bits() as usize;
        let body = crate::bodies::record_mut(world_index, body_id);
        body.sleep_velocity = sleep_velocity;
        let transient =
            crate::body::flags::IS_SPEED_CAPPED | crate::body::flags::HAD_TIME_OF_IMPACT;
        let sim_flags = sim_col.get(o + FLAGS).to_bits();
        let state_flags = flags_col.get(i * crate::body::STATE_STRIDE);
        body.body_move_index = i as i32;
        body.flags =
            (body.flags & !transient) | (state_flags & crate::body::flags::IS_SPEED_CAPPED);
        sim_col.set(
            o + FLAGS,
            f32::from_bits(sim_flags & !(transient | crate::body::flags::IS_FAST)),
        );
        flags_col.set(i * crate::body::STATE_STRIDE, state_flags & !transient);
        crate::events::write_move(world_index, i);
        let awake = !crate::continuous::sleep_enabled()
            || body.flags & crate::body::flags::ENABLE_SLEEP == 0
            || sleep_velocity > body.sleep_threshold;
        let flags = sim_col.get(o + FLAGS).to_bits() & !crate::body::flags::IS_FAST;
        sim_col.set(o + FLAGS, f32::from_bits(flags));
        let mut fast_candidate = false;
        if awake {
            body.sleep_time = 0.0;
            let max_motion = maxf(max_delta_position, max_velocity * h);
            fast_candidate = enable_continuous
                && flags_col.get(i * crate::body::STATE_STRIDE) & DYNAMIC != 0
                && max_motion > body.safety_factor * sim_col.get(o + MIN_EXTENT);
        }
        if !awake {
            body.sleep_time += h;
        }
        if fast_candidate {
            sim_col.set(
                o + FLAGS,
                f32::from_bits(flags | crate::body::flags::IS_FAST),
            );
            if flags & crate::body::flags::IS_BULLET != 0 {
                crate::continuous::add_bullet(i);
            } else if crate::continuous::callback_body(world_index, i) {
                crate::callback_work::note(
                    world_index,
                    crate::callback_work::Work::ContinuousSerial,
                );
                crate::continuous::add_deferred(i);
                continue;
            } else {
                crate::callback_work::note(
                    world_index,
                    crate::callback_work::Work::ContinuousParallel,
                );
                crate::continuous::solve(world_index, worker, i);
            }
        } else {
            sim_col.set(o + ROTATION0, q.v.x);
            sim_col.set(o + ROTATION0 + 1, q.v.y);
            sim_col.set(o + ROTATION0 + 2, q.v.z);
            sim_col.set(o + ROTATION0 + 3, q.s);
            sim_col.set(o + CENTER0, center.x);
            sim_col.set(o + CENTER0 + 1, center.y);
            sim_col.set(o + CENTER0 + 2, center.z);
        }

        finish_body(world_index, worker, i, sim_col, body_id);
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) unsafe fn finish_body(
    world_index: usize,
    worker: usize,
    i: usize,
    sim_col: Col<f32>,
    body_id: usize,
) {
    // Continuous collision can clip the rotation before inertia is rebuilt.
    let sim = read_sim(sim_col, i);
    let rotation_matrix = Mat3::from_quat(sim.rotation);
    write_sim_inv_inertia_world(
        sim_col,
        i,
        rotation_matrix
            .mul(sim.inv_inertia_local)
            .mul(rotation_matrix.transpose()),
    );
    crate::arena::mark_finalize_island(world_index, worker, body_id);
    crate::arena::refit_body(world_index, worker, sim_col, i);
}
