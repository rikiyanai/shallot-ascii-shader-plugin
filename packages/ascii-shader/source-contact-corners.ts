/**
 * Immutable source-corner correspondence compiled with a source revision.
 *
 * This is deliberately a numeric packet rather than a CPU object graph.  The
 * source worker builds the balanced tree once; render/contact code only reads
 * the packet and never rebuilds it or scans the source vertex stream.
 */
export interface SourceCornerIndexData {
  readonly schema:'shallot.source-corner-index.v1';
  readonly revision:number;
  readonly vertexCount:number;
  readonly positionCount:number;
  /** One xyz position per unique exact source position, in tree-node order. */
  readonly positions:Float32Array;
  /** Lowest original vertex index represented by each node. */
  readonly canonical:Uint32Array;
  readonly left:Int32Array;
  readonly right:Int32Array;
  readonly axis:Uint8Array;
  /** Per-node xyz bounds, used to prune tolerance queries. */
  readonly bounds:Float32Array;
}

function checkRevision(revision:number):void{
  if(!Number.isSafeInteger(revision)||revision<0)throw new RangeError('source corner index requires a non-negative safe revision');
}
function point(positions:ArrayLike<number>,vertex:number):readonly [number,number,number]{
  return [positions[vertex*3]!,positions[vertex*3+1]!,positions[vertex*3+2]!];
}
function finite(p:readonly [number,number,number]):boolean{return p.every(Number.isFinite);}
function key(p:readonly [number,number,number]):string{return p.map(v=>Object.is(v,-0)?0:v).join(',');}

interface Group{readonly position:readonly [number,number,number];readonly canonical:number}
interface Node{readonly group:Group;readonly axis:0|1|2;left?:Node;right?:Node;min:Float32Array;max:Float32Array}

function build(groups:readonly Group[]):Node|undefined{
  if(!groups.length)return;
  const min=new Float32Array(groups[0]!.position),max=new Float32Array(min);
  for(let i=1;i<groups.length;i++)for(let k=0;k<3;k++){
    min[k]=Math.min(min[k]!,groups[i]!.position[k]!);max[k]=Math.max(max[k]!,groups[i]!.position[k]!);
  }
  let axis:0|1|2=0;if(max[1]!-min[1]!>max[0]!-min[0]!)axis=1;if(max[2]!-min[2]!>max[axis]!-min[axis]!)axis=2;
  const sorted=[...groups].sort((a,b)=>a.position[axis]!-b.position[axis]!||a.canonical-b.canonical),middle=sorted.length>>1;
  const node:Node={group:sorted[middle]!,axis,min,max};node.left=build(sorted.slice(0,middle));node.right=build(sorted.slice(middle+1));return node;
}
function flatten(root:Node):SourceCornerIndexData{
  const nodes:Node[]=[];const visit=(node:Node|undefined):number=>{if(!node)return -1;const at=nodes.length;nodes.push(node);const left=visit(node.left),right=visit(node.right);(node as Node&{flatLeft?:number}).flatLeft=left;(node as Node&{flatRight?:number}).flatRight=right;return at;};
  visit(root);
  const positions=new Float32Array(nodes.length*3),canonical=new Uint32Array(nodes.length),left=new Int32Array(nodes.length),right=new Int32Array(nodes.length),axis=new Uint8Array(nodes.length),bounds=new Float32Array(nodes.length*6);
  for(let i=0;i<nodes.length;i++){const node=nodes[i]!,p=node.group.position;positions.set(p,i*3);canonical[i]=node.group.canonical;left[i]=(node as Node&{flatLeft?:number}).flatLeft??-1;right[i]=(node as Node&{flatRight?:number}).flatRight??-1;axis[i]=node.axis;bounds.set([...node.min,...node.max],i*6);}
  return {schema:'shallot.source-corner-index.v1',revision:-1,vertexCount:0,positionCount:nodes.length,positions,canonical,left,right,axis,bounds};
}

/** Compile once from an immutable source revision. */
export function compileSourceCornerIndex(revision:number,positions:ArrayLike<number>,indices?:ArrayLike<number>):SourceCornerIndexData|undefined{
  checkRevision(revision);if(positions.length%3)throw new RangeError('source corner index requires xyz source positions');
  const vertexCount=positions.length/3;if(!vertexCount)return;
  const groups=new Map<string,Group>();
  for(let at=0;at<(indices?.length??vertexCount);at++){
    const vertex=indices?indices[at]!:at;
    if(!Number.isInteger(vertex)||vertex<0||vertex>=vertexCount)throw new RangeError('source corner index requires valid original indices');
    const p=point(positions,vertex);if(!finite(p))throw new TypeError('source corner index rejects nonfinite positions');const k=key(p),old=groups.get(k);if(!old||vertex<old.canonical)groups.set(k,{position:p,canonical:vertex});
  }
  // A lower duplicate outside this source's actual indexed incidence cannot
  // be its boundary port (surface-relations vertexInDraw/vertexPort).
  if(!groups.size)return;
  const packet=flatten(build([...groups.values()])!);return {...packet,revision,vertexCount};
}

function distanceToBounds(point:readonly [number,number,number],bounds:Float32Array,at:number):number{
  let d=0;for(let k=0;k<3;k++){const low=bounds[at*6+k]!,high=bounds[at*6+3+k]!,delta=point[k]!<low?low-point[k]! : point[k]!>high?point[k]!-high:0;d+=delta*delta;}return d;
}
function distance(a:readonly [number,number,number],positions:Float32Array,at:number):number{let d=0;for(let k=0;k<3;k++){const x=a[k]!-positions[at*3+k]!;d+=x*x;}return d;}

/** Returns a canonical source vertex, or undefined for no/ambiguous matches. */
export function querySourceCorner(index:SourceCornerIndexData|undefined,point:readonly [number,number,number],tolerance:number):number|undefined{
  if(!index||!finite(point)||!Number.isFinite(tolerance)||tolerance<0)return;
  const radius=tolerance*tolerance;let found:number|undefined;let matches=0;const visit=(at:number):void=>{
    if(at<0||matches>1||distanceToBounds(point,index.bounds,at)>radius)return;
    if(distance(point,index.positions,at)<=radius){found=index.canonical[at];matches++;if(matches>1)return;}
    const axis=index.axis[at]!,delta=point[axis]!-index.positions[at*3+axis]!;
    if(delta<=0){visit(index.left[at]!);visit(index.right[at]!);}else{visit(index.right[at]!);visit(index.left[at]!);}
  };visit(0);return matches===1?found:undefined;
}
