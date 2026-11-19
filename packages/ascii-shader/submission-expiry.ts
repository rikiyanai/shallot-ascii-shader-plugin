/** Submission-clock retention. Work is proportional to entries used now or
 * actually expiring, never the complete high-water cache on every frame.
 * Disposal remains the caller's owner: clear() cancels pending callbacks. */
export class SubmissionExpiry<K extends object> {
 private frame=0;
 private readonly records=new Map<K,{bucket:number;release:()=>void}>();
 private readonly buckets:(Set<K>|undefined)[];
 constructor(readonly grace=240){
  if(!Number.isSafeInteger(grace)||grace<0)throw Error('Submission grace must be a non-negative integer');
  this.buckets=new Array(grace+1);
 }
 /** Supply release on registration. Reuse needs only the stable identity,
  * not another per-frame closure or retirement record. */
 use(key:K,release?:()=>void):void{
  const bucket=this.frame%this.buckets.length,prior=this.records.get(key);
  if(prior?.bucket===bucket)return;
  if(!prior&&!release)throw Error('New submission entry requires its release owner');
  if(prior)this.buckets[prior.bucket]?.delete(key);
  const record=prior??{bucket,release:release!};record.bucket=bucket;
  this.records.set(key,record);
  (this.buckets[bucket]??=new Set()).add(key);
 }
 /** Exactly one call at the existing submitted/discarded encoder seam.
  * An entry used in frame n retires at n+grace+1, matching age > grace. */
 submitted():void{
  this.frame++;
  const bucket=this.frame%this.buckets.length,expired=this.buckets[bucket];
  if(!expired)return;
  this.buckets[bucket]=undefined;
  for(const key of expired){const record=this.records.get(key)!;this.records.delete(key);record.release();}
 }
 clear():void{this.records.clear();this.buckets.fill(undefined);}
}
