import {cube} from '@dylanebert/shallot';
import {opposite,type Direction,type Point,type Railway,type Track,type TrainKind} from './railway';
import type {SourceConstruction,SourcePathSite,SourcePathFeature,SourcePathPair} from '../../../../packages/shallot-main/source-construction';

export type Geometry = {vertices:Float32Array;indices:Uint32Array;construction?:SourceConstruction};
export interface TrackCollider {center:Point;size:Point;yaw:number;pitch:number}
class Parts {
    readonly colliders:TrackCollider[]=[];
    private vertices:number[]=[];
    private indices:number[]=[];
    private paths:SourcePathFeature[]=[];private pairs:SourcePathPair[]=[];
    private ports=new Map<string,number>();
    private port(site:SourcePathSite):number{
        const vertex=(i:number)=>this.vertices.slice(i*8,i*8+3);
        const point=site.kind==='vertex'?vertex(site.vertex):[0,1,2].map(k=>site.barycentric.reduce((sum,w,i)=>sum+w*vertex(this.indices[site.face*3+i]!)[k]!,0));
        const key=point.map(n=>Math.round(n*1e5)).join(',');let id=this.ports.get(key);
        if(id===undefined){id=this.ports.size;this.ports.set(key,id);}return id;
    }
    /** `faces`: the ORIGINAL face ranges this path's construction owns (its
     * own ribbon strip); every other face keeps its native drawing. */
    private path(sites:SourcePathSite[],rank:number,faces?:readonly (readonly [number,number])[]):SourcePathFeature{
        const path={id:this.paths.length,sites,ports:[this.port(sites[0]!),this.port(sites.at(-1)!)] as [number,number],rank,...(faces?.length?{faces}:{})};this.paths.push(path);return path;
    }
    pair(a:SourcePathFeature,b:SourcePathFeature,gap:SourcePathSite[]):void{
        this.pairs.push({id:this.pairs.length,paths:[a.id,b.id],endPorts:[[a.ports[0],b.ports[0]],[a.ports[1],b.ports[1]]],gap});
    }
    add(g:Geometry,center:Point=[0,0,0],yaw=0,pitch=0):void {
        const base=this.vertices.length/8,face=this.indices.length/3,c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
        const rotate=(x:number,y:number,z:number):Point=>[c*x+s*(sp*y+cp*z),cp*y-sp*z,-s*x+c*(sp*y+cp*z)];
        for(let i=0;i<g.vertices.length;i+=8){
            const p=rotate(g.vertices[i]!,g.vertices[i+1]!,g.vertices[i+2]!);
            const n=rotate(g.vertices[i+4]!,g.vertices[i+5]!,g.vertices[i+6]!);
            this.vertices.push(p[0]+center[0],p[1]+center[1],p[2]+center[2],g.vertices[i+3]!,...n,g.vertices[i+7]!);
        }
        for(const i of g.indices)this.indices.push(i+base);
        if(g.construction?.paths){
            const site=(s:SourcePathSite):SourcePathSite=>s.kind==='vertex'?{kind:'vertex',vertex:s.vertex+base}:{...s,face:s.face+face};
            const byId=new Map<number,SourcePathFeature>();
            for(const path of g.construction.paths)byId.set(path.id,this.path(path.sites.map(site),path.rank,path.faces?.map(([first,count])=>[first+face,count] as const)));
            for(const pair of g.construction.pathPairs??[])this.pair(byId.get(pair.paths[0])!,byId.get(pair.paths[1])!,pair.gap.map(site));
        }
    }
    box(center:Point,size:Point,yaw=0,pitch=0):void {
        this.add(cube(size),center,yaw,pitch);this.colliders.push({center,size,yaw,pitch});
    }
    /** Connected top ribbon: no individual segment boxes, caps or uprights. */
    ribbon(points:Point[],width:number,thickness=.025,rank=0):{path:SourcePathFeature;edges:[SourcePathSite[],SourcePathSite[]]} {
        const base=this.vertices.length/8,firstFace=this.indices.length/3;
        for(let i=0;i<points.length;i++){
            const p=points[i]!,a=points[Math.max(0,i-1)]!,b=points[Math.min(points.length-1,i+1)]!;
            const d=b.map((v,k)=>v-a[k]!) as Point,length=Math.hypot(...d)||1;
            const tx=d[0]/length,ty=d[1]/length,tz=d[2]/length,h=Math.hypot(tx,tz)||1;
            for(const side of [-1,1])this.vertices.push(p[0]+side*width/2*tz/h,p[1],p[2]-side*width/2*tx/h,0,-ty*tx/h,h,-ty*tz/h,0);
            if(i){
                const a=base+(i-1)*2,b=base+i*2;
                this.indices.push(a,b,b+1,a,b+1,a+1);
                const before=points[i-1]!,delta=p.map((v,k)=>v-before[k]!) as Point;
                const center=before.map((v,k)=>(v+p[k]!)/2) as Point;center[1]-=thickness/2;
                this.colliders.push({center,size:[width,thickness,Math.hypot(...delta)+.002],yaw:Math.atan2(delta[0],delta[2]),pitch:-Math.atan2(delta[1],Math.hypot(delta[0],delta[2]))});
            }
        }
        const sites:SourcePathSite[]=points.map((_,i)=>i===points.length-1?
            {kind:'face',face:firstFace+(i-1)*2,barycentric:[0,.5,.5]}:
            {kind:'face',face:firstFace+i*2+1,barycentric:[.5,0,.5]});
        // The ribbon's own strip (two triangles per segment) is the path's face range.
        return {path:this.path(sites,rank,[[firstFace,(points.length-1)*2]]),edges:[points.map((_,i)=>({kind:'vertex',vertex:base+i*2})),points.map((_,i)=>({kind:'vertex',vertex:base+i*2+1}))]};
    }
    finish():Geometry {return {vertices:new Float32Array(this.vertices),indices:new Uint32Array(this.indices),...(this.paths.length?{construction:{schema:'shallot.source-construction.v1' as const,role:'solid' as const,components:[],paths:this.paths,pathPairs:this.pairs}}:{})};}
}

/** Low-poly cylinder along its local Z axis; shared by every steam vehicle. */
function cylinder(radius:number,length:number,sides=12):Geometry {
    const vertices:number[]=[],indices:number[]=[];
    for(let i=0;i<sides;i++){
        const a=i/sides*Math.PI*2,b=(i+1)/sides*Math.PI*2;
        const corners:Point[]=[[Math.cos(a)*radius,Math.sin(a)*radius,-length/2],[Math.cos(b)*radius,Math.sin(b)*radius,-length/2],[Math.cos(b)*radius,Math.sin(b)*radius,length/2],[Math.cos(a)*radius,Math.sin(a)*radius,length/2]];
        const base=vertices.length/8,n=[Math.cos((a+b)/2),Math.sin((a+b)/2),0];
        for(const p of corners)vertices.push(...p,0,...n,0);
        indices.push(base,base+1,base+2,base,base+2,base+3);
        for(const sign of [-1,1]){
            const start=vertices.length/8;
            for(const p of [[0,0,sign*length/2],[Math.cos(a)*radius,Math.sin(a)*radius,sign*length/2],[Math.cos(b)*radius,Math.sin(b)*radius,sign*length/2]])vertices.push(...p,0,0,0,sign,0);
            if(sign===1)indices.push(start,start+1,start+2);else indices.push(start,start+2,start+1);
        }
    }
    return {vertices:new Float32Array(vertices),indices:new Uint32Array(indices)};
}

export function trainGeometry(kind:TrainKind):Geometry {
    const p=new Parts();
    p.box([0,.14,0],[.85,.22,1.65]);
    for(const z of [-.55,.55])for(const x of [-.43,.43])p.add(cylinder(.21,.13,10),[x,0,z],Math.PI/2);
    if(kind==='steam'){
        p.add(cylinder(.32,1.05),[0,.5,.25]);
        p.box([0,.57,-.5],[.78,.68,.6]);
        p.box([0,.95,-.5],[.94,.12,.72]);
        p.add(cylinder(.1,.45,8),[0,.85,.55],0,Math.PI/2);
        p.box([0,.41,-.83],[.55,.42,.08]);
    }else {
        p.box([0,.5,0],[.82,.65,1.58]);
        p.box([0,.89,0],[kind==='metro'?.87:.94,.14,1.74]);
        // Window frames and door jambs are actual geometry in every view.
        for(const x of [-.43,.43])for(const z of [-.5,0,.5]){
            p.box([x,.66,z],[.055,.28,.035]);
            p.box([x,.52,z],[.055,.035,.38]);
        }
        if(kind==='metro')p.box([0,.55,.82],[.66,.32,.12]);
        else p.box([0,1,0],[.5,.06,1.3]);
    }
    return p.finish();
}

/** Whole track mesh is compiled on edit only, never during camera movement. */
export function trackGeometry(railway:Railway,track:Track):Geometry & {colliders:TrackCollider[]} {
    const p=new Parts(),ports=railway.ports(track),local=(point:Point):Point=>[point[0]-track.x*2,point[1]-railway.height(track.x,track.z),point[2]-track.z*2];
    const pairs:[Direction,Direction][]=[];
    if(track.kind==='turntable'){
        const axis=(track.branch%2) as Direction;pairs.push([axis,opposite(axis)]);
        p.add(cylinder(.94,.11,16),[0,.04,0],0,Math.PI/2);
        // Inscribed support under the round plate; the bridge/tie colliders
        // below retain their exact heights and rotated segment directions.
        p.colliders.push({center:[0,.04,0],size:[1.3,.11,1.3],yaw:0,pitch:0});
    }else if(ports.length<=2){const a=ports[0]??0;pairs.push([a,ports[1]??opposite(a)]);}
    else for(let i=1;i<ports.length;i++)pairs.push([ports[0]!,ports[i]!]);
    for(const [entry,exit] of pairs){
        const height=railway.height(track.x,track.z)+.15;
        const flat=opposite(entry)===exit&&railway.endpoint(track,entry)[1]===height&&railway.endpoint(track,exit)[1]===height;
        const steps=flat?1:8;
        const railPaths:ReturnType<Parts['ribbon']>[]=[];
        for(const side of [-1,1]){
            const points:Point[]=[];
            for(let i=0;i<=steps;i++){
                const pose=railway.sample(track,entry,exit,i/steps),point=local(pose.position),h=Math.hypot(pose.tangent[0],pose.tangent[2])||1;
                point[0]+=side*.3*pose.tangent[2]/h;point[2]-=side*.3*pose.tangent[0]/h;points.push(point);
            }
            railPaths.push(p.ribbon(points,.055));
            // Sparse sleeper ends OUTSIDE the gauge; no ladder across the rails.
            for(const u of [.25,.75]){
                const pose=railway.sample(track,entry,exit,u),center=local(pose.position),h=Math.hypot(pose.tangent[0],pose.tangent[2])||1;
                center[0]+=side*.49*pose.tangent[2]/h;center[2]-=side*.49*pose.tangent[0]/h;center[1]-=.06;
                p.ribbon([center.map((v,k)=>v-pose.tangent[k]!*.035) as Point,center.map((v,k)=>v+pose.tangent[k]!*.035) as Point],.2,.025,1);
            }
        }
        p.pair(railPaths[0]!.path,railPaths[1]!.path,[...railPaths[0]!.edges[1],...railPaths[1]!.edges[0].slice().reverse()]);
    }
    if(track.kind==='station'){
        p.box([.78,.19,0],[.35,.38,1.9]);
        p.box([.92,.78,0],[.08,1.15,.08]);p.box([.92,1.35,0],[.12,.32,.65]);
    }
    if(track.kind==='switch'){
        p.box([.8,.24,.7],[.1,.48,.1],0,track.branch%2?.6:-.6);
        p.box([.8,.48,.7],[.22,.1,.2]);
    }
    return {...p.finish(),colliders:p.colliders};
}


/** A static rail network is one mesh, with exact shared connector vertices.
 * Per-tile physics remains separate; turntable drawings stay on their bodies. */
export function trackNetworkGeometry(pieces:Iterable<{geometry:Geometry;position:Point}>):Geometry {
    const parts=new Parts();
    for(const {geometry,position} of pieces)parts.add(geometry,position);
    const raw=parts.finish(),vertices:number[]=[],remap=new Uint32Array(raw.vertices.length/8),known=new Map<string,number>();
    for(let i=0;i<raw.vertices.length;i+=8){
        const record=raw.vertices.subarray(i,i+8),key=record.join(',');
        let id=known.get(key);
        if(id===undefined){id=vertices.length/8;known.set(key,id);for(const value of record)vertices.push(value);}
        remap[i/8]=id;
    }
    const site=(s:SourcePathSite):SourcePathSite=>s.kind==='vertex'?{kind:'vertex',vertex:remap[s.vertex]!}:s;
    const construction=raw.construction&&{...raw.construction,paths:raw.construction.paths!.map(p=>({...p,sites:p.sites.map(site)})),pathPairs:raw.construction.pathPairs!.map(p=>({...p,gap:p.gap.map(site)}))};
    return {vertices:new Float32Array(vertices),indices:raw.indices.map(i=>remap[i]!),...(construction?{construction}:{})};
}
