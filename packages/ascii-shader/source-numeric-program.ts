
import {TOPOLOGY_HEADER_WORDS,PACKED_RUN_CONTEXT,TOPOLOGY_LIMB_MAGIC,TOPOLOGY_LIMB_MAGIC_WORD,TOPOLOGY_PART_MAGIC,TOPOLOGY_PART_MAGIC_WORD,TOPOLOGY_PART_PACKET_MAGIC,TOPOLOGY_PART_PROGRAM_WORDS} from './compiled-construction-topology.ts';

/** Registration kernels, never displayed-frame work. Hashes identify exact
 * keys; canonical labels are minimum original IDs. Lists are sorted before
 * publication. Ordered boundary/part contexts are existing compile-time
 * decisions. Runtime incidence readers expose only the three fields they
 * consume; welded components remain with the build/inspection producer. */
export const SOURCE_NUMERICS_SHADER=/* wgsl */`
struct P {dims:vec4u,facts:vec4u,core:vec4u,graph:vec4u,material:vec4u,scratch:vec4u,pose:vec4u}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> keys:array<u32>;
@group(0) @binding(2) var<storage,read> indexed:array<u32>;
@group(0) @binding(3) var<storage,read> semantic:array<u32>;
@group(0) @binding(4) var<storage,read_write> scratch:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> packet:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> program:array<atomic<u32>>;
fn load(at:u32)->u32{return atomicLoad(&packet[at]);}
fn store(at:u32,value:u32){atomicStore(&packet[at],value);}
fn csr(at:u32)->u32{return load(p.core.y+at);}
fn indexAt(h:u32)->u32{return load(p.core.x+h);}
fn nextCorner(h:u32)->u32{return h/3u*3u+(h+1u)%3u;}
fn firstEdge(e:u32)->u32{return csr(csr(7u)+e);}
fn edgeDegree(e:u32)->u32{return csr(csr(7u)+e+1u)-firstEdge(e);}
fn edgeCorner(e:u32)->u32{return csr(csr(8u)+firstEdge(e));}
fn mixBits(value:u32)->u32{var h=value;h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;return h^(h>>16u);}
fn normalizedZero(value:u32)->u32{return select(value,0u,(value&0x7fffffffu)==0u);}
fn vertexKey(v:u32)->vec3u{
 let width=p.facts.y;let base=v*width;
 var key=vec3u(keys[base],keys[base+1u],0u);if(width==3u){key=vec3u(normalizedZero(key.x),normalizedZero(key.y),normalizedZero(keys[base+2u]));}return key;
}
fn keyHash(key:vec3u)->u32{return mixBits(key.x^mixBits(key.y)^mixBits(key.z+0x9e3779b9u))&p.facts.z;}
fn insertVertex(v:u32){
 let key=vertexKey(v);var at=keyHash(key);
 for(var probe=0u;probe<=p.facts.z;probe++){
  var found=atomicLoad(&scratch[at]);
  loop{if(found!=0u){break;}let claim=atomicCompareExchangeWeak(&scratch[at],0u,v+1u);found=claim.old_value;if(claim.exchanged){return;}}
  if(all(vertexKey(found-1u)==key)){atomicMin(&scratch[at],v+1u);return;}at=(at+1u)&p.facts.z;
 }
 atomicOr(&packet[15u],1u);
}
fn findVertex(v:u32)->u32{
 let key=vertexKey(v);var at=keyHash(key);
 for(var probe=0u;probe<=p.facts.z;probe++){let found=atomicLoad(&scratch[at]);if(found!=0u&&all(vertexKey(found-1u)==key)){return found-1u;}at=(at+1u)&p.facts.z;}
 atomicOr(&packet[15u],1u);return v;
}
fn keyLess(a:vec3u,b:vec3u)->bool{return a.x<b.x||(a.x==b.x&&(a.y<b.y||(a.y==b.y&&a.z<b.z)));}
struct EdgeKey {a:vec3u,b:vec3u}
fn contactKey(e:u32)->EdgeKey{
 let h=edgeCorner(e);let a=vertexKey(indexAt(h));let b=vertexKey(indexAt(nextCorner(h)));
 if(keyLess(a,b)){return EdgeKey(a,b);}return EdgeKey(b,a);
}
fn sameEdge(a:EdgeKey,b:EdgeKey)->bool{return all(a.a==b.a)&&all(a.b==b.b);}
fn contactHash(key:EdgeKey)->u32{return mixBits(keyHash(key.a)^mixBits(keyHash(key.b)))&p.facts.z;}
/** CPU contact orientation compares decimal key strings, not numeric tuples.
 * A ':' sorts AFTER a remaining decimal digit; the final string end sorts
 * BEFORE it. Preserve that distinction for prefix values such as1 and10. */
fn decimalLess(a:u32,b:u32,colon:bool)->bool{
 if(a==b){return false;}var ad=1u;var bd=1u;
 loop{if(a/ad<10u){break;}ad*=10u;}loop{if(b/bd<10u){break;}bd*=10u;}
 loop{
  let av=(a/ad)%10u;let bv=(b/bd)%10u;if(av!=bv){return av<bv;}
  if(ad==1u||bd==1u){return select(ad==1u,bd==1u,colon);}ad/=10u;bd/=10u;
 }
 return false;
}
fn sourceForward(e:u32)->bool{
 let h=edgeCorner(e);let a=vertexKey(indexAt(h));let b=vertexKey(indexAt(nextCorner(h)));
 for(var k=0u;k<p.facts.y;k++){if(a[k]!=b[k]){return decimalLess(a[k],b[k],k+1u<p.facts.y);}}return false;
}
fn contactCanonical(e:u32)->u32{
 let key=contactKey(e);var at=contactHash(key);
 for(var probe=0u;probe<=p.facts.z;probe++){let found=atomicLoad(&scratch[at]);if(found!=0u&&sameEdge(contactKey(found-1u),key)){return found-1u;}at=(at+1u)&p.facts.z;}
 atomicOr(&packet[15u],2u);return e;
}
// Old baked six-word records must never be published under the current
// twelve-word run-program header. Unavailable metadata keeps incumbent runs;
// rebuilding the semantic asset supplies the new declared anchors.
fn currentPartPacket()->bool{
 if(arrayLength(&semantic)<35u){return false;}
 let packet=semantic[33u];let words=semantic[34u];let length=arrayLength(&semantic);
 if(packet>length||words<5u||words>length-packet){return false;}
 if(semantic[packet]!=${TOPOLOGY_PART_PACKET_MAGIC}u){return false;}
 let first=semantic[packet+1u];let count=semantic[packet+2u];let members=semantic[packet+3u];let memberCount=semantic[packet+4u];
 if(first<5u||first>words||count==0u||count>(words-first)/${TOPOLOGY_PART_PROGRAM_WORDS}u){return false;}
 return members>=first+count*${TOPOLOGY_PART_PROGRAM_WORDS}u&&members<=words&&memberCount<=words-members;
}
@compute @workgroup_size(64) fn seedRuntime(@builtin(global_invocation_id) gid:vec3u){
 let id=(gid.x+gid.y*p.pose.w*64u);
 if(id<arrayLength(&indexed)){store(p.core.y+id,indexed[id]);}
 if(id<p.dims.y){let klass=(semantic[semantic[12u]+id/16u]>>((id%16u)*2u))&3u;atomicStore(&program[TOPOLOGY_HEADER+id],klass);}
 if(id<semantic[15u]){atomicStore(&program[TOPOLOGY_HEADER+p.dims.y+id],semantic[semantic[14u]+id]);}

 // the vertex contexts, named by the run-slot limb magic (header words 8-10).
 if(id<semantic[32u]){atomicStore(&program[TOPOLOGY_HEADER+p.dims.y+semantic[15u]+id],semantic[semantic[31u]+id]);}

 // limb table. They are consumed by construction-runs through the same
 // resident program slot; visibility never changes these records.
 // Copy the opaque declared payload in parallel. Validate its VERSION only
 // once below (id==0) before publishing any reader pointer, not per edge.
 if(id<semantic[34u]){atomicStore(&program[TOPOLOGY_HEADER+p.dims.y+semantic[15u]+semantic[32u]+id],semantic[semantic[33u]+id]);}
 if(p.material.y!=0u&&id<semantic[21u]){store(p.material.y+id,semantic[semantic[20u]+id]);}
 if(id==0u){
  atomicStore(&program[0u],p.facts.w);atomicStore(&program[1u],semantic[6u]);atomicStore(&program[16u],semantic[9u]);
  atomicStore(&program[37u],PACKED_CONTEXT);atomicStore(&program[38u],TOPOLOGY_HEADER);atomicStore(&program[39u],p.dims.y);
  atomicStore(&program[40u],TOPOLOGY_HEADER+p.dims.y);atomicStore(&program[41u],semantic[15u]);
  atomicStore(&program[42u],TOPOLOGY_HEADER);atomicStore(&program[43u],p.dims.y);
  if(semantic[32u]!=0u){atomicStore(&program[${TOPOLOGY_LIMB_MAGIC_WORD}u],${TOPOLOGY_LIMB_MAGIC}u);atomicStore(&program[${TOPOLOGY_LIMB_MAGIC_WORD+1}u],TOPOLOGY_HEADER+p.dims.y+semantic[15u]);atomicStore(&program[${TOPOLOGY_LIMB_MAGIC_WORD+2}u],semantic[32u]);}
  if(currentPartPacket()){let partBase=TOPOLOGY_HEADER+p.dims.y+semantic[15u]+semantic[32u];let packet=semantic[33u];atomicStore(&program[${TOPOLOGY_PART_MAGIC_WORD}u],${TOPOLOGY_PART_MAGIC}u);atomicStore(&program[${TOPOLOGY_PART_MAGIC_WORD+1}u],partBase+semantic[packet+1u]);atomicStore(&program[${TOPOLOGY_PART_MAGIC_WORD+2}u],semantic[packet+2u]);atomicStore(&program[${TOPOLOGY_PART_MAGIC_WORD+3}u],partBase+semantic[packet+3u]);atomicStore(&program[${TOPOLOGY_PART_MAGIC_WORD+4}u],semantic[packet+4u]);}
 }
}
@compute @workgroup_size(64) fn applyBoundaryContexts(@builtin(global_invocation_id) gid:vec3u){
 let id=(gid.x+gid.y*p.pose.w*64u);if(id>=semantic[19u]/2u){return;}let at=semantic[18u]+id*2u;let h=semantic[at];let context=semantic[at+1u];
 let group=(context&0x60000000u)|((context&0x07ffffffu)<<2u);atomicOr(&program[TOPOLOGY_HEADER+h],group);
}
@compute @workgroup_size(64) fn hashVertices(@builtin(global_invocation_id) gid:vec3u){if((gid.x+gid.y*p.pose.w*64u)<p.dims.x){insertVertex((gid.x+gid.y*p.pose.w*64u));}}
@compute @workgroup_size(64) fn resolveVertices(@builtin(global_invocation_id) gid:vec3u){
 let v=(gid.x+gid.y*p.pose.w*64u);if(v<p.dims.x){store(p.core.w+v,findVertex(v));}
}
@compute @workgroup_size(64) fn linkVertices(@builtin(global_invocation_id) gid:vec3u){
 let v=(gid.x+gid.y*p.pose.w*64u);if(v>=p.dims.x){return;}let root=load(p.core.w+v);
 let old=atomicExchange(&scratch[p.scratch.x+root],v+1u);atomicStore(&scratch[p.scratch.y+v],old);
 atomicAdd(&packet[p.graph.y+root],1u);
}
@compute @workgroup_size(64) fn sortVertices(@builtin(global_invocation_id) gid:vec3u){
 let v=(gid.x+gid.y*p.pose.w*64u);if(v>=p.dims.x){return;}let root=load(p.core.w+v);var cursor=atomicLoad(&scratch[p.scratch.x+root]);var next=0xffffffffu;
 loop{if(cursor==0u){break;}let other=cursor-1u;if(other>v){next=min(next,other);}cursor=atomicLoad(&scratch[p.scratch.y+other]);}
 store(p.graph.x+v,select(next+1u,0u,next==0xffffffffu));
}
@compute @workgroup_size(64) fn hashContactEdges(@builtin(global_invocation_id) gid:vec3u){
 let e=(gid.x+gid.y*p.pose.w*64u);if(e>=p.dims.z||edgeDegree(e)!=1u){return;}let key=contactKey(e);var at=contactHash(key);
 for(var probe=0u;probe<=p.facts.z;probe++){
  var found=atomicLoad(&scratch[at]);loop{if(found!=0u){break;}let claim=atomicCompareExchangeWeak(&scratch[at],0u,e+1u);found=claim.old_value;if(claim.exchanged){return;}}
  if(sameEdge(contactKey(found-1u),key)){atomicMin(&scratch[at],e+1u);return;}at=(at+1u)&p.facts.z;
 }
 atomicOr(&packet[15u],2u);
}
@compute @workgroup_size(64) fn resolveContactEdges(@builtin(global_invocation_id) gid:vec3u){
 let e=(gid.x+gid.y*p.pose.w*64u);if(e<p.dims.z){if(edgeDegree(e)==1u){store(p.graph.w+e,contactCanonical(e)+1u);}else{store(p.graph.w+e,0u);}}
}
@compute @workgroup_size(64) fn linkContactEdges(@builtin(global_invocation_id) gid:vec3u){
 let e=(gid.x+gid.y*p.pose.w*64u);if(e>=p.dims.z){return;}let stored=load(p.graph.w+e);if(stored==0u){return;}let root=stored-1u;
 let old=atomicExchange(&scratch[p.scratch.x+root],e+1u);atomicStore(&scratch[p.scratch.y+e],old);atomicAdd(&scratch[p.scratch.z+root],1u);
}
@compute @workgroup_size(1) fn prefixContacts(){
 var group=0u;var member=0u;
 for(var e=0u;e<p.dims.z;e++){
  let count=atomicLoad(&scratch[p.scratch.z+e]);if(load(p.graph.w+e)!=e+1u||count<2u){continue;}
  atomicStore(&scratch[p.scratch.w+e],group+1u);
  if(group<p.dims.w&&member+count<=p.facts.x){store(p.graph.z+group*2u,p.material.x+member*2u);store(p.graph.z+group*2u+1u,count);}
  else{atomicOr(&packet[15u],4u);}
  member+=count;group++;
 }
 if(group!=p.dims.w||member!=p.facts.x){atomicOr(&packet[15u],4u);}
}
@compute @workgroup_size(64) fn assembleContacts(@builtin(global_invocation_id) gid:vec3u){
 let e=(gid.x+gid.y*p.pose.w*64u);if(e>=p.dims.z){return;}let stored=load(p.graph.w+e);if(stored==0u){return;}let root=stored-1u;
 let group=atomicLoad(&scratch[p.scratch.w+root]);if(group==0u){store(p.graph.w+e,0u);return;}
 if(load(15u)!=0u){return;}var rank=0u;var cursor=atomicLoad(&scratch[p.scratch.x+root]);
 loop{if(cursor==0u){break;}let other=cursor-1u;if(other<e){rank++;}cursor=atomicLoad(&scratch[p.scratch.y+other]);}
 let at=load(p.graph.z+(group-1u)*2u)+rank*2u;let h=edgeCorner(e);
 store(at,h);store(at+1u,select(0u,1u,sourceForward(e)));store(p.graph.w+e,group);
 let old=atomicLoad(&program[TOPOLOGY_HEADER+h]);atomicStore(&program[TOPOLOGY_HEADER+h],(old&3u)|0x40000000u|((group-1u)<<2u));
}
`.replaceAll('TOPOLOGY_HEADER',String(TOPOLOGY_HEADER_WORDS)+'u').replaceAll('PACKED_CONTEXT',String(PACKED_RUN_CONTEXT)+'u');

