/**
 * Source-owned, exact planar component union.
 *
 * This is deliberately a small source-revision product.  It does not infer
 * joins from distance, poses, projected bounds, or a bounding box.  A caller
 * may use a `union` result as a replacement construction only when this
 * product says that every boundary fragment is part of the same certified
 * planar union.  Anything it cannot prove is returned as `separate` so the
 * complete source contours remain available to the incumbent path.
 */

export type SourceComponentPoint = readonly [number, number, number];
export type SourceComponentContourKind = 'outer' | 'hole' | 'paper';

export interface SourceComponentContour {
  readonly feature: number;
  readonly contributors: readonly number[];
  readonly points: readonly SourceComponentPoint[];
  readonly kind: SourceComponentContourKind;
}

export interface SourceComponentIntersection {
  readonly point: SourceComponentPoint;
  readonly sourceFeatures: readonly number[];
  readonly contributors: readonly number[];
}

export interface SourceComponentUnionLoop {
  readonly kind: SourceComponentContourKind;
  readonly points: readonly SourceComponentPoint[];
  readonly sourceFeatures: readonly number[];
  readonly contributors: readonly number[];
  readonly closed: true;
}

export interface SourceComponentUnionSegment {
  readonly loop: number;
  readonly from: SourceComponentPoint;
  readonly to: SourceComponentPoint;
  readonly sourceFeatures: readonly number[];
  readonly contributors: readonly number[];
  readonly intersections: readonly SourceComponentIntersection[];
}

export interface SourceComponentFeatureAlias {
  readonly retained: number;
  readonly aliases: readonly number[];
}

export interface SourceComponentUnionResult {
  readonly status: 'union' | 'separate';
  readonly loops: readonly SourceComponentUnionLoop[];
  readonly segments: readonly SourceComponentUnionSegment[];
  readonly intersections: readonly SourceComponentIntersection[];
  readonly sourceFeatures: readonly number[];
  readonly contributors: readonly number[];
  readonly featureAliases: readonly SourceComponentFeatureAlias[];
}

type Point2 = readonly [number, number];
type Plane = { readonly origin: SourceComponentPoint; readonly normal: Point3 };
type Point3 = readonly [number, number, number];
type Split = {
  readonly t: number;
  readonly point: SourceComponentPoint;
  readonly intersections: readonly SourceComponentIntersection[];
};
type Edge = {
  readonly contour: number;
  readonly feature: number;
  readonly contributors: readonly number[];
  readonly from: SourceComponentPoint;
  readonly to: SourceComponentPoint;
  readonly from2: Point2;
  readonly to2: Point2;
  splits: Split[];
};
type Kept = {
  readonly from: SourceComponentPoint;
  readonly to: SourceComponentPoint;
  readonly sourceFeatures: readonly number[];
  readonly contributors: readonly number[];
  readonly intersections: readonly SourceComponentIntersection[];
};

const KIND_ORDER: Record<SourceComponentContourKind, number> = {
  outer: 0,
  hole: 1,
  paper: 2,
};

const add3 = (a: Point3, b: Point3): Point3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a: Point3, b: Point3): Point3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a: Point3, t: number): Point3 => [a[0] * t, a[1] * t, a[2] * t];
const dot3 = (a: Point3, b: Point3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: Point3, b: Point3): Point3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length3 = (a: Point3): number => Math.hypot(a[0], a[1], a[2]);

const finitePoint = (p: SourceComponentPoint): boolean => p.length === 3 && p.every(Number.isFinite);

const pointKey = (p: SourceComponentPoint): string => p.map((v) => Object.is(v, -0) ? 0 : v).join(',');
const pointCompare = (a: SourceComponentPoint, b: SourceComponentPoint): number => {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
};
const uniqueSortedNumbers = (values: Iterable<number>): number[] => [...new Set(values)].sort((a, b) => a - b);

function planeOf(points: readonly SourceComponentPoint[]): Plane | null {
  if (points.length < 3 || !points.every(finitePoint)) return null;
  let normal: Point3 | null = null;
  for (let i = 1; i < points.length - 1 && !normal; i++) {
    const candidate = cross3(sub3(points[i], points[0]), sub3(points[i + 1], points[0]));
    if (length3(candidate) > 0) normal = candidate;
  }
  if (!normal) return null;
  // This is source authority, not a geometric proximity test.  A contour
  // must satisfy its plane equation exactly; a very small but nonzero
  // residual is a different source plane and must keep the complete
  // incumbent rather than being welded into this construction.
  if (points.some((point) => dot3(sub3(point, points[0]), normal) !== 0)) return null;
  return { origin: points[0], normal };
}

function compatiblePlane(a: Plane, b: Plane): boolean {
  const cross = cross3(a.normal, b.normal);
  if (cross[0] !== 0 || cross[1] !== 0 || cross[2] !== 0) return false;
  return dot3(sub3(b.origin, a.origin), a.normal) === 0;
}

function projectionAxis(normal: Point3): 0 | 1 | 2 {
  const absolute = normal.map(Math.abs);
  if (absolute[1] > absolute[0] && absolute[1] >= absolute[2]) return 1;
  if (absolute[2] > absolute[0] && absolute[2] > absolute[1]) return 2;
  return 0;
}

function project(point: SourceComponentPoint, axis: 0 | 1 | 2): Point2 {
  if (axis === 0) return [point[1], point[2]];
  if (axis === 1) return [point[0], point[2]];
  return [point[0], point[1]];
}

function area2(points: readonly Point2[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2;
}

const cross2 = (a: Point2, b: Point2): number => a[0] * b[1] - a[1] * b[0];
const sub2 = (a: Point2, b: Point2): Point2 => [a[0] - b[0], a[1] - b[1]];

function interpolate(a: SourceComponentPoint, b: SourceComponentPoint, t: number): SourceComponentPoint {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function onSegment(point: Point2, a: Point2, b: Point2): boolean {
  if (cross2(sub2(point, a), sub2(b, a)) !== 0) return false;
  return point[0] >= Math.min(a[0], b[0]) && point[0] <= Math.max(a[0], b[0]) &&
    point[1] >= Math.min(a[1], b[1]) && point[1] <= Math.max(a[1], b[1]);
}

function edgeIntersections(a: Edge, b: Edge): SourceComponentPoint[] {
  const r = sub2(a.to2, a.from2);
  const s = sub2(b.to2, b.from2);
  const q = sub2(b.from2, a.from2);
  const denominator = cross2(r, s);
  if (denominator !== 0) {
    const t = cross2(q, s) / denominator;
    const u = cross2(q, r) / denominator;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      // Preserve the source endpoint object at exact endpoint hits.  This
      // avoids manufacturing a numerically different graph vertex from the
      // same source coordinate.
      return [t === 0 ? a.from : t === 1 ? a.to : interpolate(a.from, a.to, t)];
    }
    return [];
  }
  if (cross2(q, r) !== 0) return [];
  const result: SourceComponentPoint[] = [];
  const candidates: readonly [SourceComponentPoint, Point2][] = [
    [a.from, a.from2], [a.to, a.to2], [b.from, b.from2], [b.to, b.to2],
  ];
  for (const [point, projected] of candidates) {
    if (!onSegment(projected, a.from2, a.to2) || !onSegment(projected, b.from2, b.to2)) continue;
    if (!result.some((value) => pointKey(value) === pointKey(point))) result.push(point);
  }
  return result;
}

function parameter(edge: Edge, point: SourceComponentPoint): number {
  const delta = sub3(edge.to, edge.from);
  let axis = 0;
  if (Math.abs(delta[1]) > Math.abs(delta[axis])) axis = 1;
  if (Math.abs(delta[2]) > Math.abs(delta[axis])) axis = 2;
  return delta[axis] === 0 ? 0 : (point[axis] - edge.from[axis]) / delta[axis];
}

function pointLocation(point: Point2, polygon: readonly Point2[]): -1 | 0 | 1 {
  let inside = false;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    if (onSegment(point, a, b)) return 0;
    const crosses = (a[1] > point[1]) !== (b[1] > point[1]);
    if (crosses) {
      const x = a[0] + ((point[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
      if (x > point[0]) inside = !inside;
    }
  }
  return inside ? 1 : -1;
}

function simple(points: readonly Point2[]): boolean {
  if (points.length < 3 || area2(points) === 0) return false;
  for (let i = 0; i < points.length; i++) {
    const next = (i + 1) % points.length;
    if (points[i][0] === points[next][0] && points[i][1] === points[next][1]) return false;
    for (let j = i + 1; j < points.length; j++) {
      const other = (j + 1) % points.length;
      const adjacent = next === j || other === i;
      if (adjacent) continue;
      const left: Edge = {
        contour: 0, feature: 0, contributors: [],
        from: [points[i][0], points[i][1], 0], to: [points[next][0], points[next][1], 0],
        from2: points[i], to2: points[next], splits: [],
      };
      const right: Edge = {
        contour: 0, feature: 0, contributors: [],
        from: [points[j][0], points[j][1], 0], to: [points[other][0], points[other][1], 0],
        from2: points[j], to2: points[other], splits: [],
      };
      if (edgeIntersections(left, right).length > 0) return false;
    }
  }
  return true;
}

function rotateCanonical(points: readonly SourceComponentPoint[]): SourceComponentPoint[] {
  let start = 0;
  for (let i = 1; i < points.length; i++) if (pointCompare(points[i], points[start]) < 0) start = i;
  return [...points.slice(start), ...points.slice(0, start)];
}

function sortedUniquePoints(points: readonly SourceComponentPoint[]): SourceComponentPoint[] {
  const byKey = new Map<string, SourceComponentPoint>();
  for (const point of points) byKey.set(pointKey(point), point);
  return [...byKey.values()].sort(pointCompare);
}

function sortedUnique(values: Iterable<number>): number[] {
  return uniqueSortedNumbers(values);
}

function normalizedLoop(contour: SourceComponentContour, axis: 0 | 1 | 2): SourceComponentUnionLoop {
  let points = [...contour.points];
  const area = area2(points.map((point) => project(point, axis)));
  const wantsCounterClockwise = contour.kind === 'outer';
  if ((area > 0) !== wantsCounterClockwise) points.reverse();
  points = rotateCanonical(points);
  return {
    kind: contour.kind,
    points,
    sourceFeatures: [contour.feature],
    contributors: sortedUnique(contour.contributors),
    closed: true,
  };
}

function resultForSeparate(
  contours: readonly SourceComponentContour[],
  axis: 0 | 1 | 2 = 2,
): SourceComponentUnionResult {
  const loops = [...contours]
    .sort((a, b) => a.feature - b.feature || KIND_ORDER[a.kind] - KIND_ORDER[b.kind])
    .map((contour) => normalizedLoop(contour, axis));
  return {
    status: 'separate',
    loops,
    segments: [],
    intersections: [],
    sourceFeatures: sortedUnique(contours.map((contour) => contour.feature)),
    contributors: sortedUnique(contours.flatMap((contour) => contour.contributors)),
    featureAliases: [],
  };
}

function provenanceFor(
  points: readonly SourceComponentIntersection[],
): SourceComponentIntersection[] {
  const byKey = new Map<string, SourceComponentIntersection>();
  for (const value of points) {
    const key = `${pointKey(value.point)}|${value.sourceFeatures.join(',')}|${value.contributors.join(',')}`;
    byKey.set(key, {
      point: value.point,
      sourceFeatures: sortedUnique(value.sourceFeatures),
      contributors: sortedUnique(value.contributors),
    });
  }
  return [...byKey.values()].sort((a, b) => pointCompare(a.point, b.point));
}

function aliases(features: readonly number[]): SourceComponentFeatureAlias[] {
  const sorted = sortedUnique(features);
  if (sorted.length < 2) return [];
  return [{ retained: sorted[0], aliases: sorted.slice(1) }];
}

function loopArea(loop: SourceComponentUnionLoop, axis: 0 | 1 | 2): number {
  return Math.abs(area2(loop.points.map((point) => project(point, axis))));
}

/**
 * Union exact coplanar closed outer contours while retaining holes/paper as
 * protected loops.  The output is source-coordinate data and is independent
 * of camera, pose, projected size, and glyph selection.
 */
export function unionSourceComponentContours(
  contours: readonly SourceComponentContour[],
): SourceComponentUnionResult {
  if (contours.length === 0) {
    return {
      status: 'separate', loops: [], segments: [], intersections: [],
      sourceFeatures: [], contributors: [], featureAliases: [],
    };
  }
  const featureSet = new Set<number>();
  for (const contour of contours) {
    if (!Number.isInteger(contour.feature) || featureSet.has(contour.feature)) return resultForSeparate(contours);
    featureSet.add(contour.feature);
    if (contour.points.length < 3 || contour.points.some((point) => !finitePoint(point))) return resultForSeparate(contours);
  }
  const plane = planeOf(contours[0].points);
  if (!plane || contours.some((contour) => {
    const candidate = planeOf(contour.points);
    return candidate === null || !compatiblePlane(plane, candidate);
  })) return resultForSeparate(contours);

  const axis = projectionAxis(plane.normal);
  const canonical = contours.map((contour) => ({
    input: contour,
    projected: contour.points.map((point) => project(point, axis)),
  }));
  if (canonical.some(({ input, projected }) => !simple(projected))) return resultForSeparate(contours, axis);

  const bySourceOrder = (a: typeof canonical[number], b: typeof canonical[number]): number =>
    a.input.feature - b.input.feature || KIND_ORDER[a.input.kind] - KIND_ORDER[b.input.kind];
  const outer = canonical.filter(({ input }) => input.kind === 'outer').sort(bySourceOrder).map((item) => {
    if (area2(item.projected) > 0) return item;
    const points = [...item.input.points].reverse();
    return { input: { ...item.input, points }, projected: points.map((point) => project(point, axis)) };
  });
  const protectedContours = canonical.filter(({ input }) => input.kind !== 'outer').sort(bySourceOrder);
  if (outer.length === 0) return resultForSeparate(contours, axis);

  // Holes and paper are protected source geometry.  A boundary collision is
  // not a safe union: preserve every original loop instead.
  for (const protectedContour of protectedContours) {
    for (const outerContour of outer) {
      for (let i = 0; i < protectedContour.projected.length; i++) {
        const a = protectedContour.projected[i];
        const b = protectedContour.projected[(i + 1) % protectedContour.projected.length];
        for (let j = 0; j < outerContour.projected.length; j++) {
          const c = outerContour.projected[j];
          const d = outerContour.projected[(j + 1) % outerContour.projected.length];
          const left: Edge = { contour: 0, feature: 0, contributors: [], from: [a[0], a[1], 0], to: [b[0], b[1], 0], from2: a, to2: b, splits: [] };
          const right: Edge = { contour: 0, feature: 0, contributors: [], from: [c[0], c[1], 0], to: [d[0], d[1], 0], from2: c, to2: d, splits: [] };
          if (edgeIntersections(left, right).length > 0) return resultForSeparate(contours, axis);
        }
      }
    }
  }

  const edges: Edge[] = [];
  for (let contourIndex = 0; contourIndex < outer.length; contourIndex++) {
    const item = outer[contourIndex];
    const points = item.input.points;
    for (let i = 0; i < points.length; i++) {
      const from = points[i];
      const to = points[(i + 1) % points.length];
      edges.push({
        contour: contourIndex,
        feature: item.input.feature,
        contributors: sortedUnique(item.input.contributors),
        from,
        to,
        from2: project(from, axis),
        to2: project(to, axis),
        splits: [
          { t: 0, point: from, intersections: [] },
          { t: 1, point: to, intersections: [] },
        ],
      });
    }
  }

  const allIntersections: SourceComponentIntersection[] = [];
  let interacted = false;
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      if (edges[i].contour === edges[j].contour) continue;
      const points = edgeIntersections(edges[i], edges[j]);
      if (points.length === 0) continue;
      interacted = true;
      for (const point of points) {
        const intersection: SourceComponentIntersection = {
          point,
          sourceFeatures: sortedUnique([edges[i].feature, edges[j].feature]),
          contributors: sortedUnique([...edges[i].contributors, ...edges[j].contributors]),
        };
        allIntersections.push(intersection);
        edges[i].splits.push({ t: parameter(edges[i], point), point, intersections: [intersection] });
        edges[j].splits.push({ t: parameter(edges[j], point), point, intersections: [intersection] });
      }
    }
  }
  // Containment is also an interaction even when no boundary edges cross.
  for (let i = 0; i < outer.length; i++) {
    for (let j = i + 1; j < outer.length; j++) {
      const sample = outer[i].projected[0];
      if (pointLocation(sample, outer[j].projected) === 1 || pointLocation(outer[j].projected[0], outer[i].projected) === 1) interacted = true;
    }
  }

  const projectedPolygons = outer.map((item) => item.projected);
  const kept: Kept[] = [];
  for (const edge of edges) {
    const byPoint = new Map<string, Split>();
    for (const split of edge.splits) {
      const key = pointKey(split.point);
      const existing = byPoint.get(key);
      if (!existing || split.intersections.length > existing.intersections.length) byPoint.set(key, split);
    }
    const splits = [...byPoint.values()].sort((a, b) => a.t - b.t || pointCompare(a.point, b.point));
    for (let i = 0; i < splits.length - 1; i++) {
      const from = splits[i];
      const to = splits[i + 1];
      if (pointKey(from.point) === pointKey(to.point)) continue;
      const midpoint3 = scale3(add3(from.point, to.point), 0.5);
      const midpoint = project(midpoint3, axis);
      let covered = false;
      for (let polygon = 0; polygon < projectedPolygons.length; polygon++) {
        if (polygon === edge.contour) continue;
        const location = pointLocation(midpoint, projectedPolygons[polygon]);
        // A fragment touching another boundary is an internal seam or an
        // overlapping boundary.  Keeping either would double-draw the join.
        if (location !== -1) {
          covered = true;
          break;
        }
      }
      if (covered) continue;
      const records = provenanceFor([...from.intersections, ...to.intersections]);
      kept.push({
        from: from.point,
        to: to.point,
        sourceFeatures: [edge.feature],
        contributors: edge.contributors,
        intersections: records,
      });
    }
  }

  if (!interacted || kept.length === 0) return resultForSeparate(contours, axis);

  const outgoing = new Map<string, Kept[]>();
  const incoming = new Map<string, number>();
  for (const segment of kept) {
    const fromKey = pointKey(segment.from);
    const toKey = pointKey(segment.to);
    const list = outgoing.get(fromKey) ?? [];
    list.push(segment);
    outgoing.set(fromKey, list);
    incoming.set(toKey, (incoming.get(toKey) ?? 0) + 1);
  }
  // A branching graph is not a certified single-loop construction.  Keep
  // the full children instead of guessing a bridge or a winding.
  if ([...outgoing.values()].some((list) => list.length !== 1) || [...incoming.values()].some((count) => count !== 1)) {
    return resultForSeparate(contours, axis);
  }

  const remaining = [...kept].sort((a, b) => pointCompare(a.from, b.from) || pointCompare(a.to, b.to));
  const used = new Set<Kept>();
  const loops: SourceComponentUnionLoop[] = [];
  const outputSegments: SourceComponentUnionSegment[] = [];
  while (used.size < remaining.length) {
    const first = remaining.find((segment) => !used.has(segment));
    if (!first) break;
    const startKey = pointKey(first.from);
    let current: Kept | undefined = first;
    const loopSegments: Kept[] = [];
    const points: SourceComponentPoint[] = [];
    while (current && !used.has(current)) {
      used.add(current);
      loopSegments.push(current);
      points.push(current.from);
      const next = outgoing.get(pointKey(current.to));
      current = next?.[0];
      if (current && pointKey(current.from) === startKey) break;
    }
    if (!current || pointKey(loopSegments[loopSegments.length - 1].to) !== startKey) return resultForSeparate(contours, axis);
    const sourceFeatures = sortedUnique(loopSegments.flatMap((segment) => segment.sourceFeatures));
    const contributors = sortedUnique(loopSegments.flatMap((segment) => segment.contributors));
    const loopIndex = loops.length;
    loops.push({ kind: 'outer', points: rotateCanonical(points), sourceFeatures, contributors, closed: true });
    for (const segment of loopSegments) {
      outputSegments.push({
        loop: loopIndex,
        from: segment.from,
        to: segment.to,
        sourceFeatures: sortedUnique(segment.sourceFeatures),
        contributors: sortedUnique(segment.contributors),
        intersections: provenanceFor(segment.intersections),
      });
    }
  }

  const protectedLoops = protectedContours.map((item) => normalizedLoop(item.input, axis));
  const ordered = [
    ...loops.map((loop, oldIndex) => ({ loop, oldIndex })),
    ...protectedLoops.map((loop) => ({ loop, oldIndex: undefined })),
  ].sort((a, b) => KIND_ORDER[a.loop.kind] - KIND_ORDER[b.loop.kind] || pointCompare(a.loop.points[0], b.loop.points[0]));
  const loopRemap = new Map<number, number>();
  ordered.forEach((entry, newIndex) => {
    if (entry.oldIndex !== undefined) loopRemap.set(entry.oldIndex, newIndex);
  });
  const allLoops = ordered.map((entry) => entry.loop);
  const remappedSegments = outputSegments.map((segment) => ({
    ...segment,
    loop: loopRemap.get(segment.loop) ?? segment.loop,
  }));
  // Assign fully covered source contours to the smallest containing output
  // loop so a disjoint loop is never aliased into a different construction.
  const featuresByLoop = new Map<number, number[]>();
  for (const item of outer) {
    const sample = project(item.input.points[0], axis);
    let selected: { readonly index: number; readonly area: number } | undefined;
    for (let oldIndex = 0; oldIndex < loops.length; oldIndex++) {
      const loop = loops[oldIndex];
      const location = pointLocation(sample, loop.points.map((point) => project(point, axis)));
      if (location === -1) continue;
      const candidate = { index: loopRemap.get(oldIndex) ?? oldIndex, area: loopArea(loop, axis) };
      if (!selected || candidate.area < selected.area || (candidate.area === selected.area && candidate.index < selected.index)) selected = candidate;
    }
    if (selected) {
      const group = featuresByLoop.get(selected.index) ?? [];
      group.push(item.input.feature);
      featuresByLoop.set(selected.index, group);
    }
  }
  const featureAliases = [...featuresByLoop.values()]
    .flatMap((features) => aliases(features))
    .sort((a, b) => a.retained - b.retained);
  const sourceFeatures = sortedUnique(contours.map((contour) => contour.feature));
  const contributors = sortedUnique(contours.flatMap((contour) => contour.contributors));
  return {
    status: 'union',
    loops: allLoops,
    segments: remappedSegments,
    intersections: provenanceFor(allIntersections),
    sourceFeatures,
    contributors,
    featureAliases,
  };
}
