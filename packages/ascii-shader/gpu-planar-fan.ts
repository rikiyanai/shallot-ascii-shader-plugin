/** GPU playback of PlaneCapFan's coupled native axes (not its CPU fitter).
 * Source incidence establishes the three faces. One width/depth/height state
 * determines every corner and repeat count, including the opposite return.
 * Other incidence/view contexts keep the incumbent hard-boundary producer.
 * References and native interfaces are documented in plane-cap-fan.ts.
 */
export const GPU_PLANAR_FAN=/* wgsl */`
struct NativeFan {
 valid:u32, mirror:f32, origin:vec2i, width:i32, depth:i32, wall:i32,
 vertices:array<Point,7>,
}

// search is REMOVED). The fan's incidence is its box packet's ORIGINAL edge
// descriptors: one whole pair per source edge with the original vertex IDs in
// visible.z, retained whether or not the edge is occluded (visibility is the
// head point's flag, separate from the geometry). Every descriptor counts,
// occluded or visible, so the seven vertices, the coupled width/depth/wall and
// every repeat count are the same under ANY occlusion. An absent (zeroed) or
// malformed descriptor means the source graph is incomplete: the complete
// incumbent keeps the box; no edge is ever inferred. Occlusion clips units
// only in fanWord, where an occluded edge emits nothing.
struct FanGraph {vertices:array<Point,7>,adjacency:array<vec3u,7>,degree:array<u32,7>,vertexCount:u32,edgeCount:u32,ok:bool}
fn nativeFanGraph(run:Run)->FanGraph{
 var g:FanGraph;g.ok=false;
 var vertices:array<Point,7>;var adjacency:array<vec3u,7>;var degree:array<u32,7>;
 var vertexCount=0u;var edgeCount=0u;
 for(var i=0u;i+1u<run.range.y;i+=2u){
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];
  // An absent descriptor (no projected geometry) is an incomplete source graph.
  if(a.visible.y==0u||b.visible.y==0u){return g;}
  if(a.visible.z==NONE||b.visible.z==NONE||a.visible.z==b.visible.z){return g;}
  var pair=vec2u(0);
  for(var side=0u;side<2u;side++){
   var q=a;if(side==1u){q=b;}
   var found=NONE;
   for(var v=0u;v<vertexCount;v++){if(vertices[v].visible.z==q.visible.z){found=v;}}
   if(found==NONE){if(vertexCount==7u){return g;}found=vertexCount;vertices[vertexCount]=q;vertexCount++;}
   pair[side]=found;
  }
  for(var side=0u;side<2u;side++){
   let v=pair[side];let other=pair[1u-side];
   if(degree[v]>=3u){return g;}
   for(var n=0u;n<degree[v];n++){if(adjacency[v][n]==other){return g;}}
   adjacency[v][degree[v]]=other;degree[v]++;
  }
  edgeCount++;
 }
 g.vertices=vertices;g.adjacency=adjacency;g.degree=degree;g.vertexCount=vertexCount;g.edgeCount=edgeCount;g.ok=true;return g;
}
fn nativeFanOf(vertexIn:array<Point,7>,adjacencyIn:array<vec3u,7>,degreeIn:array<u32,7>)->NativeFan{
 var fan:NativeFan;fan.valid=0u;
 var vertices=vertexIn;var adjacency=adjacencyIn;var degree=degreeIn;
 // Removing the sole three-way interior corner leaves one exterior cycle.
 // Actual original vertex IDs, not projected proximity, establish adjacency.
 var center=NONE;
 for(var c=0u;c<7u;c++){
  if(degree[c]!=3u){continue;}var matches=true;
  for(var v=0u;v<7u;v++){
   if(v==c){continue;}var remaining=degree[v];
   for(var n=0u;n<degree[v];n++){if(adjacency[v][n]==c){remaining--;}}
   matches=matches&&remaining==2u;
  }
  if(matches){if(center!=NONE){return fan;}center=c;}
 }
 if(center==NONE){return fan;}
 var arms=adjacency[center];
 for(var i=0u;i<3u;i++){for(var j=i+1u;j<3u;j++){
  if(vertices[arms[j]].position.y<vertices[arms[i]].position.y){let t=arms[i];arms[i]=arms[j];arms[j]=t;}
 }}
 let upper=arms[0];let left=arms[1];let lower=arms[2];let c=vertices[center].position.xy;
 let mirror=select(-1.0,1.0,vertices[left].position.x<c.x);
 if((vertices[upper].position.x-c.x)*mirror<=0.0||vertices[upper].position.y>=c.y||vertices[lower].position.y<=c.y){return fan;}
 var between=vec3u(NONE);
 // upper-left, upper-lower, left-lower, excluding the shared center.
 let pairs=array<vec2u,3>(vec2u(upper,left),vec2u(upper,lower),vec2u(left,lower));
 for(var p=0u;p<3u;p++){
  let a=pairs[p].x;let b=pairs[p].y;
  for(var i=0u;i<degree[a];i++){let v=adjacency[a][i];if(v==center){continue;}
   for(var j=0u;j<degree[b];j++){if(adjacency[b][j]==v){if(between[p]!=NONE){return fan;}between[p]=v;}}
  }
  if(between[p]==NONE){return fan;}
 }
 if(between.x==between.y||between.x==between.z||between.y==between.z){return fan;}
 let order=array<u32,7>(between.x,upper,left,center,between.y,between.z,lower);
 for(var i=0u;i<7u;i++){fan.vertices[i]=vertices[order[i]];}
 let targetWidth=(c.x-vertices[left].position.x)*mirror;
 let width=34+16*max(0,i32(round((targetWidth-34.0)/16.0)));
 let depth=max(1,i32(round((c.y-vertices[upper].position.y)/17.0)));
 let wall=max(1,i32(round((vertices[lower].position.y-c.y)/17.0)));
 let extra=16*(depth-1);
 let native=array<vec2i,7>(vec2i(12+extra,15),vec2i(width+16+extra,15),
  vec2i(2,17*depth+12),vec2i(width+2,17*depth+14),vec2i(width+19+extra,17*(wall+1)),
  vec2i(2,17*(depth+wall)+15),vec2i(width+4,17*(depth+wall)+15));
 var offset=vec2f(0);
 for(var i=0u;i<7u;i++){offset+=fan.vertices[i].position.xy*vec2f(mirror,1)-vec2f(native[i]);}
 offset/=7.0;let origin=vec2i(i32(round(offset.x)),i32(round(offset.y/17.0)));
 // One declared native growth family; no search for a different fit. The
 // incumbent handles views outside this family's one-row displacement.
 for(var i=0u;i<7u;i++){
  let depicted=vec2f(vec2i(origin.x,origin.y*17))+vec2f(native[i]);
  if(distance(depicted,fan.vertices[i].position.xy*vec2f(mirror,1))>17.0){return fan;}
 }
 fan.valid=1u;fan.mirror=mirror;fan.origin=origin;fan.width=width;fan.depth=depth;fan.wall=wall;return fan;
}
fn nativeFan(run:Run)->NativeFan{
 var fan:NativeFan;fan.valid=0u;
 let g=nativeFanGraph(run);if(!g.ok||g.vertexCount!=7u||g.edgeCount!=9u){return fan;}
 return nativeFanOf(g.vertices,g.adjacency,g.degree);
}
fn fanWord(id:u32,run:Run,fan:NativeFan,start:u32,end:u32,unit:u32,word:u32,x:i32,row:i32,t:f32,priority:u32){
 let fromID=fan.vertices[start].visible.z;let toID=fan.vertices[end].visible.z;
 for(var i=0u;i+1u<run.range.y;i+=2u){
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];
  let forward=a.visible.z==fromID&&b.visible.z==toID;let backward=b.visible.z==fromID&&a.visible.z==toID;
  if(!forward&&!backward){continue;}
  // Occlusion clips here, per edge: the construction above never changes.
  if(!pointVisible(a)){return;}
  let advance=words[word*WS+1u];let q=interpolatePoint(a,b,select(1.0-t,t,forward));
  let pen=select(-fan.origin.x-x-i32(advance),fan.origin.x+x,fan.mirror>0.0);
  sourceWord(id,unit,q,pen,fan.origin.y+row,word,words[word*WS],advance,priority,a,b);return;
 }
}
fn fanUnits(fan:NativeFan,x:i32,step:i32,count:i32,width:u32)->vec2i{
 let pen=select(-fan.origin.x-x-i32(width),fan.origin.x+x,fan.mirror>0.0);
 return vec2i(pageUnits(pen,select(-step,step,fan.mirror>0.0),u32(max(0,count)),width));
}
fn fanRows(fan:NativeFan,first:i32,last:i32)->vec2i{
 return pageRows(vec2i(first,last)+vec2i(fan.origin.y))-vec2i(fan.origin.y);
}
fn planarFanConstruction(id:u32,run:Run,priority:u32)->bool{
 let fan=nativeFan(run);if(fan.valid==0u){return false;}
 let record=grammar[7]+12u+7u*8u;
 let upright=grammar[record];let fill=grammar[record+2u];let corner=grammar[record+3u];
 let rise=grammar[record+4u];let fall=grammar[record+5u];let short=grammar[record+7u];
 if(words[upright*WS+1u]!=4u||words[fill*WS+1u]!=16u||words[short*WS+1u]!=5u||words[corner*WS+1u]!=16u){return false;}
 let m=(fan.width-34)/16;let extra=16*(fan.depth-1);let s=fan.width+2;let base=fan.depth+fan.wall;
 let diagonal=select(fall,rise,fan.mirror>0.0);
 // The back rail grows by one whole full-width unit per front/base unit.
 fanWord(id,run,fan,0u,1u,0u,short,12+extra,0,0.0,priority);
 let backUnits=fanUnits(fan,17+extra,16,m,16u);
 for(var k=backUnits.x;k<backUnits.y;k++){fanWord(id,run,fan,0u,1u,u32(k+1),fill,17+extra+16*k,0,f32(5+16*k)/f32(fan.width+4),priority);}
 for(var k=0;k<7;k++){fanWord(id,run,fan,0u,1u,u32(m+k+1),short,17+extra+16*m+5*k,0,f32(5+16*m+5*k)/f32(fan.width+4),priority);}
 let leftRows=fanRows(fan,1,fan.depth-1);
 for(var row=leftRows.x;row<=leftRows.y;row++){fanWord(id,run,fan,0u,2u,u32(row),diagonal,16*(fan.depth-row),row,f32(row)/f32(fan.depth),priority);}
 fanWord(id,run,fan,0u,2u,u32(fan.depth),select(fall,corner,fan.mirror>0.0),0,fan.depth,1.0,priority);
 for(var k=0;k<4;k++){fanWord(id,run,fan,2u,3u,u32(k),short,16+5*k,fan.depth,f32(14+5*k)/f32(fan.width),priority);}
 let frontUnits=fanUnits(fan,36,16,m,16u);
 for(var k=frontUnits.x;k<frontUnits.y;k++){fanWord(id,run,fan,2u,3u,u32(k+4),fill,36+16*k,fan.depth,f32(34+16*k)/f32(fan.width),priority);}
 let topRows=fanRows(fan,1,fan.depth);
 for(var row=topRows.x;row<=topRows.y;row++){fanWord(id,run,fan,3u,1u,u32(row),diagonal,s+16*(fan.depth-row),row,1.0-f32(row)/f32(fan.depth),priority);}
 let backRows=fanRows(fan,1,fan.wall);
 for(var row=backRows.x;row<=backRows.y;row++){
  fanWord(id,run,fan,1u,4u,u32(row),upright,s+16+extra,row,f32(row)/f32(fan.wall),priority);
 }
 let wallRows=fanRows(fan,fan.depth+1,fan.depth+fan.wall);
 for(var absolute=wallRows.x;absolute<=wallRows.y;absolute++){
  let row=absolute-fan.depth;
  fanWord(id,run,fan,2u,5u,u32(row),upright,1,fan.depth+row,f32(row)/f32(fan.wall),priority);
  fanWord(id,run,fan,3u,6u,u32(row),upright,s+1,fan.depth+row,f32(row)/f32(fan.wall),priority);
 }
 let baseUnits=fanUnits(fan,5,16,m+2,16u);
 for(var k=baseUnits.x;k<baseUnits.y;k++){fanWord(id,run,fan,5u,6u,u32(k),fill,5+16*k,base,f32(3+16*k)/f32(fan.width+2),priority);}
 let returnRows=fanRows(fan,fan.wall+1,fan.wall+fan.depth);
 for(var absolute=returnRows.x;absolute<=returnRows.y;absolute++){let row=absolute-fan.wall;fanWord(id,run,fan,6u,4u,u32(row),diagonal,s+5+16*(fan.depth-row),fan.wall+row,1.0-f32(row)/f32(fan.depth),priority);}
 return true;
}
`;
