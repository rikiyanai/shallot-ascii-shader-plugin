import {relationPoint,type PosedRelation} from './posed-graph.ts';

export interface SourceSupport {readonly relation:PosedRelation;readonly parameter:number;readonly depth:number;readonly distance:number}
interface Edge {relation:PosedRelation;order:number;a:readonly number[];b:readonly number[];length:number;box:readonly [number,number,number,number]}
interface Node {box:readonly [number,number,number,number];edges?:readonly Edge[];left?:Node;right?:Node}
const distance=(x:number,y:number,box:Node['box'])=>Math.max(box[0]-x,0,x-box[2])**2+Math.max(box[1]-y,0,y-box[3])**2;
function tree(edges:Edge[]):Node{
 const box:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
 for(const edge of edges){box[0]=Math.min(box[0],edge.box[0]);box[1]=Math.min(box[1],edge.box[1]);box[2]=Math.max(box[2],edge.box[2]);box[3]=Math.max(box[3],edge.box[3]);}
 if(edges.length<=8)return {box,edges};
 const axis=box[2]-box[0]>=box[3]-box[1]?0:1;
 edges.sort((a,b)=>(a.box[axis]+a.box[axis+2])-(b.box[axis]+b.box[axis+2])||a.order-b.order);
 const middle=edges.length>>>1;return {box,left:tree(edges.slice(0,middle)),right:tree(edges.slice(middle))};
}

/** Exact geometric correspondence on an ALREADY selected source path. This
 * index never sees a glyph, image, vocabulary or fit score. It replaces the
 * repeated full-path scan used for each native ink/paper query, retaining the
 * original first-source tie rule and perspective parameter calculation. */
export class SourceSupportIndex {
 private readonly paths=new WeakMap<readonly PosedRelation[],Node>();
 nearest(x:number,y:number,relations:readonly PosedRelation[]):SourceSupport|undefined{
  if(!relations.length)return;
  let root=this.paths.get(relations);
  if(!root){const edges:Edge[]=[];
   for(const [order,relation]of relations.entries()){
    const a=relationPoint(relation,0),b=relationPoint(relation,1),length=(b[0]-a[0])**2+(b[1]-a[1])**2;if(!Number.isFinite(length))continue;
    edges.push({relation,order,a,b,length,box:[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]});
   }
   root=tree(edges);this.paths.set(relations,root);
  }
  let best:SourceSupport|undefined,order=Infinity;const pending:Node[]=[root];
  while(pending.length){const node=pending.pop()!;if(best&&distance(x,y,node.box)>best.distance)continue;
   if(node.edges)for(const edge of node.edges){
    const {a,b,relation,length}=edge,dx=b[0]!-a[0]!,dy=b[1]!-a[1]!,s=Math.max(0,Math.min(1,length?((x-a[0]!)*dx+(y-a[1]!)*dy)/length:0));
    const distance=(x-a[0]!-s*dx)**2+(y-a[1]!-s*dy)**2;
    if(best&&(distance>best.distance||distance===best.distance&&edge.order>=order))continue;
    const denominator=relation.b[3]*(1-s)+relation.a[3]*s;order=edge.order;
    best={relation,parameter:denominator?s*relation.a[3]/denominator:s,depth:a[2]!*(1-s)+b[2]!*s,distance};
   }else{
    const left=node.left!,right=node.right!;
    if(distance(x,y,left.box)<=distance(x,y,right.box))pending.push(right,left);else pending.push(left,right);
   }
  }
  return best;
 }
}
