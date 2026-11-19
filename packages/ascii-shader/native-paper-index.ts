import {nativePaperContains,type NativePaper} from './native-page.ts';

interface Node {
 readonly paper:NativePaper;readonly start:number;readonly end:number;readonly order:number;
 height:number;maximum:number;left?:Node;right?:Node;
}
const height=(node:Node|undefined)=>node?.height??0;
function update(node:Node):Node{
 node.height=1+Math.max(height(node.left),height(node.right));
 node.maximum=Math.max(node.end,node.left?.maximum??-Infinity,node.right?.maximum??-Infinity);return node;
}
function left(node:Node):Node{const root=node.right!;node.right=root.left;root.left=update(node);return update(root);}
function right(node:Node):Node{const root=node.left!;node.left=root.right;root.right=update(node);return update(root);}
function insert(root:Node|undefined,node:Node):Node{
 if(!root)return node;
 if(node.start<root.start||node.start===root.start&&node.order<root.order)root.left=insert(root.left,node);
 else root.right=insert(root.right,node);
 update(root);
 if(height(root.left)-height(root.right)>1){if(height(root.left!.right)>height(root.left!.left))root.left=left(root.left!);return right(root);}
 if(height(root.right)-height(root.left)>1){if(height(root.right!.left)>height(root.right!.right))root.right=right(root.right!);return left(root);}
 return root;
}
function contains(node:Node|undefined,x:number,y:number):boolean{
 if(!node||node.maximum<=x)return false;
 if(contains(node.left,x,y))return true;
 if(node.start>x)return false;
 if(node.end>x&&nativePaperContains(node.paper,x,y))return true;
 return contains(node.right,x,y);
}

/** Exact lookup of existing protected-paper constraints. The index does not
 * create, shrink or erase paper, rasterize a glyph, or choose a construction.
 * Row buckets and augmented AVL intervals only avoid visiting rectangles
 * that cannot contain the queried native point. visibleAt remains the same
 * same-pose source predicate used by NativePage's final verification. */
export class NativePaperIndex {
 private readonly rows:(Node|undefined)[];
 private readonly width:number;private readonly height:number;private order=0;
 constructor(width:number,height:number,papers:Iterable<NativePaper>=[]){
  this.width=width;this.height=height;this.rows=Array(Math.ceil(height/17));
  for(const paper of papers)this.add(paper);
 }
 add(paper:NativePaper):void{
  const [x,y,w,h]=paper.rect;
  if(![x,y,w,h].every(Number.isFinite)||w<=0||h<=0)throw Error(`Invalid protected paper ${paper.id}`);
  const order=this.order++,begin=Math.max(0,Math.floor(y/17)),end=Math.min(this.rows.length,Math.ceil((y+h)/17));
  if(x>=this.width||x+w<=0)return;
  for(let row=begin;row<end;row++)this.rows[row]=insert(this.rows[row],{paper,start:x,end:x+w,order,height:1,maximum:x+w});
 }
 contains(x:number,y:number):boolean{
  return x>=0&&x<this.width&&y>=0&&y<this.height&&contains(this.rows[Math.floor(y/17)],x,y);
 }
}
