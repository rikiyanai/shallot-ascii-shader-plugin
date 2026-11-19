import {RELATION_WORDS} from './source-coordinates.ts';

/** Source addresses and their projected pose are distinct. Visibility never
 * rewrites incidence. A smooth contour endpoint belongs to a migrating source
 * edge crossing, while a hard endpoint belongs to a source vertex/contact. */
export interface PosedPort {readonly id:string;readonly homogeneous:readonly [number,number,number,number];readonly members:readonly string[]}
export interface FeatureInterval {readonly lo:number;readonly hi:number;readonly cause:'visible'|'occluded'|'outside-view';readonly foreground?:boolean}
export interface PosedRelation {
  readonly id:string;readonly owner:string;readonly kind:'boundary'|'smooth-contour'|'crease'|'suggestive-contour';
  readonly a:readonly [number,number,number,number];readonly b:readonly [number,number,number,number];
  readonly ports:readonly [string,string];readonly regions:readonly [string,string|undefined];
  readonly intervals:readonly FeatureInterval[];
  readonly components?:readonly number[];
  /** Actual posed world-up edge, before perspective projection. */
  readonly upright?:boolean;
  /** Actual posed adjacent-triangle support, in this native camera's page.
   * Rest-mesh centroids cannot orient a material-deformed/GPU-owned contour. */
  readonly frontFacePoint?:readonly [number,number];
  /** Intrinsic rest-source chart endpoints; not native-page/run progress. */
  readonly sourceChart?:readonly [number,number];
  /** Authored adjacent face class; absent when there is no face table. */
  readonly materialClass?:number;
}
export interface PosedOwner {readonly camera:string;readonly domain:number;readonly kind:'entity'|'producer-row';readonly index:number;readonly lifetime:number}
/** One source-owner address for decoder, mesh compiler and tone attachment.
 * Raw eid and lifetime are deliberately distinct; generation is not lost. */
export function posedOwnerId(owner:Pick<PosedOwner,'camera'|'domain'|'kind'|'lifetime'>):string{
 return `${owner.camera}/${owner.domain}/${owner.kind}/${owner.lifetime}`;
}
export interface PosedGraph {readonly ports:ReadonlyMap<string,PosedPort>;readonly relations:readonly PosedRelation[];readonly owners?:ReadonlyMap<string,PosedOwner>;
 /** Complete source-owner snapshot, including wholly hidden rows. These are
  * context, not publication participants: retiring a hidden row must not
  * cancel an unrelated visible drawing. Resident source topology stays live
  * under its own actual lifetime/revision authority. */
 readonly sourceOwners?:ReadonlyMap<string,PosedOwner>}

export function relationPoint(relation:Pick<PosedRelation,'a'|'b'>,t:number):readonly [number,number,number]{
  const {a,b}=relation,w=a[3]*(1-t)+b[3]*t;
  return [0,1,2].map(axis=>(a[axis]!*(1-t)+b[axis]!*t)/w) as unknown as readonly [number,number,number];
}
/** Convert affine screen/native-chart progress to the source edge parameter.
 * Perspective does not preserve equal screen and source intervals. */
export function relationParameter(relation:Pick<PosedRelation,'a'|'b'>,s:number):number{
 const denominator=relation.b[3]*(1-s)+relation.a[3]*s;
 return denominator?s*relation.a[3]/denominator:s;
}
export function relationRange(relation:PosedRelation,from:string,range:readonly [number,number]):readonly [number,number]{
 const screen=relation.ports[0]===from?range:[1-range[1],1-range[0]];
 return [relationParameter(relation,screen[0]!),relationParameter(relation,screen[1]!)];
}

/** Decode one camera's complete relation stream and interval partition. IDs
 * depend only on source/owner/ports, never screen coordinates or clipped ends.
 * Coincident source candidates are checked in the posed graph; incompatible
 * smooth normal crossings keep separate incidences instead of forced joins. */
export function decodePosedGraph(memory:ArrayBuffer,capacity:number,intervalCapacity:number,camera:string,
  rect:{x:number;y:number;width:number;height:number},sourceWidth:number,sourceHeight:number,ownerLifetimes?:ReadonlyMap<number,number>,component?:(domain:number,face:number)=>number|undefined,
  occupied?:ReadonlyMap<number,ReadonlySet<number>>):PosedGraph{
  if(memory.byteLength<16+capacity*RELATION_WORDS*4+intervalCapacity*16)throw Error('Incomplete posed graph readback');
  const status=new Uint32Array(memory,0,4);if(status[1])throw Error(`Source graph/visibility overflow ${status[1]}`);
  const intervalCount=status[0]!,relationCount=status[2]!;
  if(intervalCount>intervalCapacity||relationCount>capacity)throw Error('Invalid posed graph record count');
  const words=new Uint32Array(memory,16,capacity*RELATION_WORDS),values=new Float32Array(memory,16,capacity*RELATION_WORDS);
  const intervalWords=new Uint32Array(memory,16+capacity*RELATION_WORDS*4),intervalFloats=new Float32Array(memory,16+capacity*RELATION_WORDS*4);
  // Validate the COMPLETE packed partition, but do not inflate wholly hidden
  // owners into hundreds of MB of strings/ports/interval objects. An owner
  // seen by EITHER a source edge or any occupied surface pixel retains ALL
  // of its original relations, including its hidden paths and source phase.
  // Original source complexes/GPU geometry are retained independently.
  const visible=occupied?new Map([...occupied].map(([key,rows])=>[key,new Set(rows)])):undefined;
  const offsets=new Uint32Array(relationCount+1);
  for(let i=0;i<intervalCount;i++){
    const at=i*4,id=intervalWords[at]!,flags=intervalWords[at+3]!,cause=flags&3,lo=intervalFloats[at+1]!,hi=intervalFloats[at+2]!;
    if(id>=relationCount||cause>2||(flags&~7)!==0||!Number.isFinite(lo)||!Number.isFinite(hi)||lo<0||hi>1||hi<=lo)throw Error('Invalid source visibility partition');
    offsets[id+1]++;
    if(visible&&cause===0){const header=id*RELATION_WORDS,key=(words[header+8]!|(words[header+10]!&0x80000000))>>>0,rows=visible.get(key)??new Set<number>();rows.add(words[header+9]!);visible.set(key,rows);}
  }
  for(let i=1;i<offsets.length;i++)offsets[i]!+=offsets[i-1]!;
  const cursors=offsets.slice(),order=new Uint32Array(intervalCount);
  for(let i=0;i<intervalCount;i++){const id=intervalWords[i*4]!;order[cursors[id]++]=i;}
  const compareIntervals=(a:number,b:number)=>intervalFloats[a*4+1]!-intervalFloats[b*4+1]!||a-b;
  const ports=new Map<string,PosedPort>(),relations:PosedRelation[]=[],owners=new Map<string,PosedOwner>(),sourceOwners=new Map<string,PosedOwner>();
  const identities=new Map<number,Map<number,{id:string;owner:PosedOwner;ports:Map<number,string>}>>();
  const candidates=new Map<string,{member:string;parameter:number;point:readonly [number,number,number,number];index:number;end:0|1}[]>();
  const attach=(id:string,point:readonly [number,number,number,number],parameter:number,member:string,index:number,end:0|1)=>{
    if(!point.every(Number.isFinite)||!Number.isFinite(parameter))throw Error('Nonfinite homogeneous source port');
    const group=candidates.get(id)??[];group.push({member,parameter,point,index,end});candidates.set(id,group);return id;
  };
  for(let i=0;i<relationCount;i++){
    const at=i*RELATION_WORDS,domain=words[at+8]!,entity=words[at+9]!,tag=words[at+10]!,kind=tag&0x1fffffff,address=words[at+11]!;
    if(kind>3)throw Error('Unknown source relation kind');
    const producer=(tag&0x80000000)!==0;
    const key=(domain|(tag&0x80000000))>>>0;
    let rows=identities.get(key);if(!rows){rows=new Map();identities.set(key,rows);}
    let identity=rows.get(entity);if(!identity){const lifetime=producer?entity:ownerLifetimes?.get(entity)??entity,owner:PosedOwner={camera,domain,kind:producer?'producer-row':'entity',index:entity,lifetime};
     identity={id:posedOwnerId(owner),owner,ports:new Map()};rows.set(entity,identity);sourceOwners.set(identity.id,owner);}
    // Even an unmaterialized hidden owner's packed partition and homogeneous
    // records must be valid. Corruption cannot masquerade as invisibility.
    for(let k=0;k<8;k++)if(!Number.isFinite(values[at+k]))throw Error('Nonfinite homogeneous source port');
    if(!Number.isFinite(values[at+16])||!Number.isFinite(values[at+17]))throw Error('Nonfinite homogeneous source port');
    if(tag&0x40000000&&(!Number.isFinite(values[at+18])||!Number.isFinite(values[at+19])))throw Error('Nonfinite posed face support');
    const indices=order.subarray(offsets[i]!,offsets[i+1]!);if(indices.length>1)indices.sort(compareIntervals);
    let end=0;for(const index of indices){const lo=intervalFloats[index*4+1]!,hi=intervalFloats[index*4+2]!;if(Math.abs(lo-end)>0.00001)throw Error('Visibility lost part of its source relation');end=hi;}
    if(Math.abs(end-1)>0.00001)throw Error('Visibility failed to partition a complete source relation');
    if(visible&&!visible.get(key)?.has(entity))continue;
    owners.set(identity.id,identity.owner);
    const owner=identity.id,id=`${owner}/${kind}/${address}`;
    const point=(offset:number):readonly [number,number,number,number]=>[
      (values[offset]!*0.5+values[offset+3]!*0.5)*rect.width+rect.x*values[offset+3]!,(-values[offset+1]!*0.5+values[offset+3]!*0.5)*rect.height+rect.y*values[offset+3]!,values[offset+2]!,values[offset+3]!];
    const a=point(at),b=point(at+4),parts:FeatureInterval[]=Array.from(indices,index=>{const at=index*4,flags=intervalWords[at+3]!;return {lo:intervalFloats[at+1]!,hi:intervalFloats[at+2]!,cause:['visible','occluded','outside-view'][flags&3] as FeatureInterval['cause'],...(flags&4?{foreground:true}:{})};});
    // A radial-curvature crossing and a normal-zero crossing may share a
    // mesh edge without being the same feature or the same source point.
    const port=(address:number)=>{if(kind===3)return `${owner}/suggestive/port-${address}`;let id=identity!.ports.get(address);if(id===undefined){id=`${owner}/port-${address}`;identity!.ports.set(address,id);}return id;};
    const atRelation=relations.length;
    const pa=attach(port(words[at+12]!),a,values[at+16]!,id,atRelation,0);
    const pb=attach(port(words[at+13]!),b,values[at+17]!,id,atRelation,1);
    const left=words[at+14]!,right=words[at+15]!;
    const components=[...new Set([component?.(domain,left),right===0xffffffff?undefined:component?.(domain,right)].filter((value):value is number=>value!==undefined))];
    const frontFacePoint:readonly [number,number]|undefined=tag&0x40000000?
      [(values[at+18]!*.5+.5)*rect.width+rect.x,(-values[at+19]!*.5+.5)*rect.height+rect.y]:undefined;
    if(frontFacePoint&&!frontFacePoint.every(Number.isFinite))throw Error('Nonfinite posed face support');
    const sourceChart=(Math.trunc(values[at+23]!)&1)!==0?[values[at+20]!,values[at+21]!] as const:undefined;
    const materialClass=values[at+22]!==0?values[at+22]!-1:undefined;
    relations.push({id,owner,kind:['boundary','smooth-contour','crease','suggestive-contour'][kind] as PosedRelation['kind'],a,b,ports:[pa,pb],regions:[`${owner}/face-${left}`,right===0xffffffff?undefined:`${owner}/face-${right}`],intervals:parts,components,frontFacePoint,sourceChart,materialClass,upright:!!(tag&0x20000000)});
  }
  // Group source-edge crossings in their own invariant parameter domain.
  // Screen coincidences cannot merge different depths or crossing parameters.
  for(const [id,members]of candidates){
    members.sort((a,b)=>a.parameter-b.parameter||a.member.localeCompare(b.member)||a.end-b.end);
    const groups:typeof members[]=[];
    for(const member of members){const prior=groups.at(-1);if(prior&&Math.abs(prior[0]!.parameter-member.parameter)<=0.00001)prior.push(member);else groups.push([member]);}
    for(const group of groups){
      group.sort((a,b)=>a.member.localeCompare(b.member)||a.end-b.end);
      const key=groups.length===1?id:`${id}/crossing-${group[0]!.member}`;
      ports.set(key,{id:key,homogeneous:group[0]!.point,members:group.map(m=>m.member)});
      for(const member of group){const relation=relations[member.index]!;const endpoints:[string,string]=[...relation.ports];endpoints[member.end]=key;relations[member.index]={...relation,ports:endpoints};}
    }
  }
  return {ports,relations:relations.sort((a,b)=>a.id.localeCompare(b.id)),owners,sourceOwners};
}
