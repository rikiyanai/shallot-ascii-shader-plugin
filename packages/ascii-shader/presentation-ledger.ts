


















export interface PresentationSnapshotLease {
  /** Release this submission's ownership. Safe to call more than once. */
  release(): void;
}

type SnapshotReleaseHandler<Snapshot extends object> = (snapshot: Snapshot) => void;

export class PresentationLedger<Canvas extends object> {
  private serial = 0;
  private requestEpoch = 0;
  private disposed = false;
  private demands: {epoch: number; resolve: () => void; reject: (error: Error) => void}[] = [];
  private readonly canvasSerials = new WeakMap<Canvas, number>();
  private readonly targetSerials = new Map<number, number>();
  /**
   * A resident snapshot can be used by more than one submitted frame. Keep a
   * count per identity until every owning frame fence settles; a serial guard
   * alone cannot protect a candidate that is not displayed yet.
   */
  private readonly snapshotRefs = new Map<object, number>();
  private readonly snapshotReleaseHandlers = new Map<object, SnapshotReleaseHandler<object>>();
  private readonly snapshotLeases = new Set<PresentationSnapshotLease>();
  /** A read/export consumer's demand for one fresh presentation. */
  demand(): Promise<void> {
    if (this.disposed) return Promise.reject(Error('MAIN presentation ledger is disposed'));
    const epoch = ++this.requestEpoch;
    return new Promise((resolve, reject) => this.demands.push({epoch, resolve, reject}));
  }
  /** True while a demand waits for a presented frame. */
  get demanded(): boolean {return this.demands.length > 0;}
  /** At encode: this frame's serial and the newest request it can answer. */
  encode(): {readonly serial: number; readonly epoch: number} {return {serial: ++this.serial, epoch: this.requestEpoch};}
  /** At fence fulfilment: may this encode publish canvas `canvas`? Newest serial wins. */
  commitCanvas(canvas: Canvas, serial: number): boolean {
    if (this.disposed || serial < (this.canvasSerials.get(canvas) ?? 0)) return false;
    this.canvasSerials.set(canvas, serial);
    return true;
  }
  /** At fence fulfilment: may this encode publish target `target`? Newest serial wins. */
  commitTarget(target: number, serial: number): boolean {
    if (this.disposed || serial < (this.targetSerials.get(target) ?? 0)) return false;
    this.targetSerials.set(target, serial);
    return true;
  }
  /**
   * Retain the candidate snapshots encoded by one submission. The handler is
   * called once per snapshot when its final submission lease is released,
   * including when disposal tears down pending submissions. The iterable is
   * identity-deduplicated because one submission may present one snapshot to
   * multiple targets.
   */
  retainSnapshots<Snapshot extends object>(snapshots: Iterable<Snapshot>, onReleased: SnapshotReleaseHandler<Snapshot>): PresentationSnapshotLease {
    const unique=[...new Set(snapshots)];
    if (this.disposed || unique.length===0) return {release():void{}};
    const refs=unique as object[];
    for (const snapshot of refs) {
      this.snapshotRefs.set(snapshot,(this.snapshotRefs.get(snapshot)??0)+1);
      // One ledger belongs to one presentation state. Keep the first handler
      // for a shared identity so its final refcount invokes release exactly
      // once even when later submissions supply a fresh closure.
      if (!this.snapshotReleaseHandlers.has(snapshot)) this.snapshotReleaseHandlers.set(snapshot,onReleased as SnapshotReleaseHandler<object>);
    }
    let released=false;
    const lease:PresentationSnapshotLease={release:():void=>{
      if (released) return;
      released=true;
      this.snapshotLeases.delete(lease);
      for (const snapshot of refs) {
        const count=(this.snapshotRefs.get(snapshot)??0)-1;
        if (count>0) { this.snapshotRefs.set(snapshot,count); continue; }
        this.snapshotRefs.delete(snapshot);
        const handler=this.snapshotReleaseHandlers.get(snapshot);
        this.snapshotReleaseHandlers.delete(snapshot);
        handler?.(snapshot);
      }
    }};
    this.snapshotLeases.add(lease);
    return lease;
  }
  /** True while at least one submitted frame still owns this snapshot. */
  isSnapshotRetained(snapshot: object): boolean { return (this.snapshotRefs.get(snapshot)??0)>0; }
  /** At fence fulfilment of a frame that presented a canvas: answer the demands it can answer. */
  answer(epoch: number): void {
    if (this.disposed) return;
    const answered = this.demands.filter(demand => demand.epoch <= epoch);
    this.demands = this.demands.filter(demand => demand.epoch > epoch);
    for (const demand of answered) demand.resolve();
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const demand of this.demands.splice(0)) demand.reject(Error('MAIN world disposed before presentation'));
    for (const lease of [...this.snapshotLeases]) lease.release();
  }
}
