import {vehicleShape,tenderShape,steamCouplers,type VehicleShape} from './vehicle-shape';
import {TrainTrail,coupledPose,type RailPose} from './train-trail';
export type Direction = 0 | 1 | 2 | 3;
export type TrackKind = 'track' | 'switch' | 'station' | 'turntable';
export type TrainKind = 'steam' | 'passenger' | 'metro';
export type Cell = readonly [number, number];
export type Point = [number, number, number];
export const TILE = 2;
export const TRAIN_RAIL_LIFT = .3;
export const DIRECTIONS: readonly Cell[] = [[0,-1],[1,0],[0,1],[-1,0]];
export const opposite = (d: Direction): Direction => ((d+2)%4) as Direction;
export const cellKey = (x: number, z: number): string => `${x},${z}`;
export interface Track {
    x: number; z: number; kind: TrackKind; ports: Set<Direction>; branch: number; turn:number;
}
export interface Train {
    id: number; kind: TrainKind; x: number; z: number; entry: Direction;
    exit: Direction; progress: number; speed: number; wait: number;
    guided: boolean; blocked:boolean; position: Point; tangent: Point;
    trailer?:RailPose;trail?:TrainTrail;
}

type TrainCheckpoint = Pick<Train,'x'|'z'|'entry'|'exit'|'progress'|'wait'|'position'|'tangent'|'trailer'>;
function separated(dx:number,dz:number,x:number,z:number,ax:number,az:number,bx:number,bz:number,af:number,bf:number,aw:number,bw:number):boolean {
    const extent=af*Math.abs(x*ax+z*az)+bf*Math.abs(x*bx+z*bz)
        +aw*Math.abs(-x*az+z*ax)+bw*Math.abs(-x*bz+z*bx);
    return Math.abs(dx*x+dz*z)>=extent+.03;
}
/** Same footprint as the engine hull, with a small stopping clearance. */
function hullsOverlap(a:RailPose,b:RailPose,as:VehicleShape,bs:VehicleShape):boolean {
    const al=Math.hypot(a.tangent[0],a.tangent[2])||1,bl=Math.hypot(b.tangent[0],b.tangent[2])||1;
    const ax=a.tangent[0]/al,az=a.tangent[2]/al,bx=b.tangent[0]/bl,bz=b.tangent[2]/bl;
    const ay=Math.abs(a.tangent[1]),by=Math.abs(b.tangent[1]);
    const ah=as.halfHeight*al+as.halfLength*ay,bh=bs.halfHeight*bl+bs.halfLength*by;
    const centerY=a.position[1]+as.offset*al-b.position[1]-bs.offset*bl;
    if(Math.abs(centerY)>=ah+bh+.03)return false;
    const dx=b.position[0]-a.position[0]+as.offset*ax*a.tangent[1]-bs.offset*bx*b.tangent[1];
    const dz=b.position[2]-a.position[2]+as.offset*az*a.tangent[1]-bs.offset*bz*b.tangent[1];
    const af=as.halfLength*al+as.halfHeight*ay,bf=bs.halfLength*bl+bs.halfHeight*by;
    return !separated(dx,dz,ax,az,ax,az,bx,bz,af,bf,as.halfWidth,bs.halfWidth)&&!separated(dx,dz,-az,ax,ax,az,bx,bz,af,bf,as.halfWidth,bs.halfWidth)
        &&!separated(dx,dz,bx,bz,ax,az,bx,bz,af,bf,as.halfWidth,bs.halfWidth)&&!separated(dx,dz,-bz,bx,ax,az,bx,bz,af,bf,as.halfWidth,bs.halfWidth);
}
function bodiesOverlap(a:Pick<Train,'position'|'tangent'|'kind'|'trailer'>,b:Pick<Train,'position'|'tangent'|'kind'|'trailer'>):boolean {
    const ap:[RailPose,VehicleShape][]=[[a,vehicleShape(a.kind)]],bp:[RailPose,VehicleShape][]=[[b,vehicleShape(b.kind)]];
    if(a.trailer)ap.push([a.trailer,tenderShape]);if(b.trailer)bp.push([b.trailer,tenderShape]);
    return ap.some(([p,s])=>bp.some(([q,t])=>hullsOverlap(p,q,s,t)));
}

/** Owns rail connectivity, not rendering or free-body physics. The editor and
 * simulation cross this same interface. Coordinates are integer tile centres. */
export class Railway {
    readonly tracks = new Map<string, Track>();
    readonly trains = new Map<number, Train>();
    readonly dirty = new Set<string>();
    readonly heights = new Map<string, number>();
    running = true;
    private serial = 0;
    private readonly starts=new Map<number,TrainCheckpoint>();
    private readonly active:Train[]=[];
    private readonly stopped=new Set<number>();
    private readonly rotating=new Set<number>();
    private readonly turnStarts=new Map<Track,number>();
    private remember(train:Train):void {
        let saved=this.starts.get(train.id);
        if(!saved){saved={x:0,z:0,entry:0,exit:0,progress:0,wait:0,position:[0,0,0],tangent:[0,0,0]};this.starts.set(train.id,saved);}
        saved.x=train.x;saved.z=train.z;saved.entry=train.entry;saved.exit=train.exit;saved.progress=train.progress;saved.wait=train.wait;
        for(let i=0;i<3;i++){saved.position[i]=train.position[i]!;saved.tangent[i]=train.tangent[i]!;}
        saved.trailer=train.trailer?{position:[...train.trailer.position],tangent:[...train.trailer.tangent]}:undefined;
    }
    private stop(train:Train):void {
        const saved=this.starts.get(train.id)!;
        train.x=saved.x;train.z=saved.z;train.entry=saved.entry;train.exit=saved.exit;train.progress=saved.progress;train.wait=saved.wait;
        for(let i=0;i<3;i++){train.position[i]=saved.position[i]!;train.tangent[i]=saved.tangent[i]!;}
        train.trailer=saved.trailer?{position:[...saved.trailer.position],tangent:[...saved.trailer.tangent]}:undefined;
        train.blocked=true;this.stopped.add(train.id);
        const track=this.tracks.get(cellKey(train.x,train.z)),turn=track&&this.turnStarts.get(track);
        if(track&&turn!==undefined){
            track.turn=turn;
            for(const other of this.active){
                const original=this.starts.get(other.id);
                if(!this.stopped.has(other.id)&&this.rotating.has(other.id)&&original?.x===track.x&&original.z===track.z)this.stop(other);
            }
        }
    }
    removeTrain(id:number):void {this.trains.delete(id);this.starts.delete(id);}
    constructor(readonly radius = 6) {}
    contains(x: number, z: number): boolean {
        return Number.isInteger(x)&&Number.isInteger(z)&&x>=-this.radius&&x<this.radius&&z>=-this.radius&&z<this.radius;
    }
    height(x: number,z: number): number { return this.contains(x,z)?this.heights.get(cellKey(x,z))??0:-1; }
    setHeight(x: number,z: number,height: number): boolean {
        if(!this.contains(x,z)||!Number.isInteger(height)||height < -1||height>4)return false;
        if(this.height(x,z)===height)return false;
        this.heights.set(cellKey(x,z),height);
        if(height===-1)this.removeTrack(x,z);
        this.changed(x,z);
        for(const d of [0,1,2,3] as Direction[]){
            const [dx,dz]=DIRECTIONS[d]!,n=this.tracks.get(cellKey(x+dx,z+dz));
            if(n&&Math.abs(height-this.height(n.x,n.z))>1)this.unlink(x,z,d);
        }
        this.connectNeighbours(x,z);
        return true;
    }
    private changed(x: number,z: number): void {
        this.dirty.add(cellKey(x,z));
        for(const [dx,dz] of DIRECTIONS)this.dirty.add(cellKey(x+dx,z+dz));
    }
    private capacity(track: Track): number { return track.kind==='switch'?3:track.kind==='turntable'?4:2; }
    private alignTurntable(track:Track):void {
        if(track.kind==='turntable'&&track.turn===0&&track.ports.size&&![...track.ports].some(d=>d%2===track.branch%2))track.branch=this.ports(track)[0]!%2;
    }
    private unlink(x: number,z: number,d: Direction): void {
        const [dx,dz]=DIRECTIONS[d]!;
        this.tracks.get(cellKey(x,z))?.ports.delete(d);
        this.tracks.get(cellKey(x+dx,z+dz))?.ports.delete(opposite(d));
        this.changed(x,z);
    }
    private connectNeighbours(x: number,z: number): void {
        const t=this.tracks.get(cellKey(x,z));if(!t)return;
        for(const d of [0,1,2,3] as Direction[]){
            const [dx,dz]=DIRECTIONS[d]!,n=this.tracks.get(cellKey(x+dx,z+dz));
            if(!n||t.ports.has(d)||Math.abs(this.height(x,z)-this.height(n.x,n.z))>1)continue;
            if(t.ports.size>=this.capacity(t)||n.ports.size>=this.capacity(n))continue;
            t.ports.add(d);n.ports.add(opposite(d));this.alignTurntable(n);this.changed(x,z);
        }
        this.alignTurntable(t);
    }
    placeTrack(x: number,z: number,kind: TrackKind='track'): boolean {
        if(!this.contains(x,z)||this.height(x,z)<0)return false;
        const key=cellKey(x,z),old=this.tracks.get(key);
        if(old){
            if(old.kind===kind)return false;
            old.kind=kind;
            while(old.ports.size>this.capacity(old))this.unlink(x,z,[...old.ports].at(-1)!);
        }else this.tracks.set(key,{x,z,kind,ports:new Set(),branch:0,turn:0});
        this.connectNeighbours(x,z);this.changed(x,z);return true;
    }
    removeTrack(x: number,z: number): boolean {
        const t=this.tracks.get(cellKey(x,z));if(!t)return false;
        for(const d of [...t.ports])this.unlink(x,z,d);
        this.tracks.delete(cellKey(x,z));this.changed(x,z);return true;
    }
    erase(x: number,z: number): void {
        if(!this.removeTrack(x,z))this.setHeight(x,z,this.height(x,z)-1);
    }
    toggle(x: number,z: number): boolean {
        const t=this.tracks.get(cellKey(x,z));if(!t||!['switch','turntable'].includes(t.kind))return false;
        if(t.turn>0)return false;
        t.branch++;if(t.kind==='turntable')t.turn=1;this.changed(x,z);return true;
    }
    moveTrack(from:Cell,to:Cell):boolean {
        const track=this.tracks.get(cellKey(...from));
        if(!track||!this.contains(...to)||this.height(...to)<0||this.tracks.has(cellKey(...to)))return false;
        if([...this.trains.values()].some(train=>train.guided&&train.x===from[0]&&train.z===from[1]))return false;
        const kind=track.kind;this.removeTrack(...from);return this.placeTrack(...to,kind);
    }
    ports(t: Track): Direction[] { return [...t.ports].sort((a,b)=>a-b); }
    exits(t: Track,entry: Direction): Direction[] {
        const ports=this.ports(t);
        if(t.kind==='turntable'){
            const axis=t.branch%2;
            return ports.filter(d=>d!==entry&&d%2===axis&&entry%2===axis);
        }
        const exits=ports.filter(d=>d!==entry);
        if(!exits.length)return [];
        return [exits[t.branch%exits.length]!];
    }
    endpoint(t: Track,d: Direction): Point {
        const [dx,dz]=DIRECTIONS[d]!,height=this.height(t.x,t.z),next=this.height(t.x+dx,t.z+dz);
        const linked=t.ports.has(d)&&next>=0;
        return [t.x*TILE+dx*TILE/2,(linked?Math.max(height,next):height)+.15,t.z*TILE+dz*TILE/2];
    }
    sample(t: Track,entry: Direction,exit: Direction,progress: number): {position: Point;tangent: Point} {
        const a=this.endpoint(t,entry),b=this.endpoint(t,exit),c:Point=[t.x*TILE,this.height(t.x,t.z)+.15,t.z*TILE];
        const u=Math.max(0,Math.min(1,progress)),v=1-u;
        const position=a.map((p,i)=>v*v*p+2*v*u*c[i]!+u*u*b[i]!) as Point;
        const derivative=a.map((p,i)=>2*v*(c[i]!-p)+2*u*(b[i]!-c[i]!)) as Point;
        const length=Math.hypot(...derivative)||1;
        return {position,tangent:derivative.map(v=>v/length) as Point};
    }
    private seedTrailer(train:Train):void {
        if(train.kind!=='steam')return;
        const trail=train.trail??new TrainTrail();trail.clear();train.trail=trail;
        const points:Point[]=[];let x=train.x,z=train.z,entry=train.entry,exit=train.exit,u=train.progress;
        // Seed a short real route behind the spawn, not a straight tail through
        // the island. A buffer end extrapolates only the missing past length.
        for(let count=0;count<10;count++){
            const track=this.tracks.get(cellKey(x,z));if(!track)break;
            for(let p=u;p>=0;p-=.025)points.push(this.sample(track,entry,exit,p).position);
            points.push(this.sample(track,entry,exit,0).position);
            const [dx,dz]=DIRECTIONS[entry]!,previous=this.tracks.get(cellKey(x+dx,z+dz));if(!previous)break;
            const toHead=opposite(entry),from=this.ports(previous).find(d=>d!==toHead);if(from===undefined)break;
            x=previous.x;z=previous.z;entry=from;exit=toHead;u=1;
        }
        for(const p of points.reverse())trail.append(p);trail.append(train.position);this.followTrailer(train);
    }
    private followTrailer(train:Train):void {
        if(train.trail)train.trailer=coupledPose(train,train.trail.sample(steamCouplers.front[2]!-steamCouplers.rear[2]!,train),steamCouplers.rear,steamCouplers.front);
    }
    addTrain(kind: TrainKind,x: number,z: number): Train|undefined {
        const t=this.tracks.get(cellKey(x,z));if(!t||t.ports.size<1)return;
        const ports=this.ports(t),entry=ports[0]!,exit=this.exits(t,entry)[0]??opposite(entry);
        const pose=this.sample(t,entry,exit,.5);
        const train:Train={id:++this.serial,kind,x,z,entry,exit,progress:.5,speed:kind==='metro'?2.5:1.8,wait:0,guided:true,blocked:false,...pose};
        this.seedTrailer(train);
        if([...this.trains.values()].some(other=>other.guided&&bodiesOverlap(train,other)))return;
        this.trains.set(train.id,train);return train;
    }
    rerail(train: Train,x: number,z: number): boolean {
        const t=this.tracks.get(cellKey(x,z));if(!t||!t.ports.size)return false;
        const entry=this.ports(t)[0]!,exit=this.exits(t,entry)[0]??opposite(entry),pose=this.sample(t,entry,exit,.5);
        const candidate={...train,...pose,x,z,entry,exit,progress:.5,trail:undefined,trailer:undefined};this.seedTrailer(candidate);
        if([...this.trains.values()].some(other=>other.id!==train.id&&other.guided&&bodiesOverlap(candidate,other)))return false;
        Object.assign(train,{x,z,entry,exit,progress:.5,wait:0,guided:true,blocked:false,trail:candidate.trail,trailer:candidate.trailer},pose);return true;
    }
    derail(train: Train): void { train.guided=false;train.blocked=false; }
    step(dt: number): void {
        if(!Number.isFinite(dt)||dt<0)throw Error('Railway step must be a finite nonnegative duration');
        if(!this.running)return;
        const steps=Math.max(1,Math.ceil(dt*60));
        for(let i=0;i<steps;i++)this.advance(dt/steps);
    }
    private advance(dt:number):void {
        this.active.length=0;this.stopped.clear();this.rotating.clear();this.turnStarts.clear();
        for(const train of this.trains.values())if(train.guided){train.blocked=false;this.remember(train);this.active.push(train);}
        const rotating=this.rotating;
        for(const track of this.tracks.values())if(track.turn>0){
            this.turnStarts.set(track,track.turn);
            const delta=Math.min(dt,track.turn);track.turn=Math.max(0,track.turn-dt);
            const a=-delta*Math.PI/2,c=Math.cos(a),s=Math.sin(a);
            for(const train of this.trains.values())if(train.guided&&train.x===track.x&&train.z===track.z){
                rotating.add(train.id);
                const dx=train.position[0]-track.x*TILE,dz=train.position[2]-track.z*TILE;
                train.position[0]=track.x*TILE+c*dx+s*dz;train.position[2]=track.z*TILE-s*dx+c*dz;
                const [tx,ty,tz]=train.tangent;train.tangent=[c*tx+s*tz,ty,-s*tx+c*tz];
                if(track.turn===0){train.entry=((train.entry+1)%4) as Direction;train.exit=((train.exit+1)%4) as Direction;}
            }
        }
        for(const train of this.trains.values()){
            if(!train.guided||rotating.has(train.id))continue;
            let track=this.tracks.get(cellKey(train.x,train.z));
            if(!track||!track.ports.has(train.exit)){this.derail(train);continue;}
            if(train.wait>0){train.wait=Math.max(0,train.wait-dt);continue;}
            // Fixed-step caller; length is 2 for straight and about 1.62 for a
            // quarter turn. Progress remains local when crossing a connector.
            const length=opposite(train.entry)===train.exit?TILE:1.623;
            train.progress+=dt*train.speed/length;
            while(train.progress>=1&&train.guided){
                const old=track,exit=train.exit,[dx,dz]=DIRECTIONS[exit]!;
                const next=this.tracks.get(cellKey(old.x+dx,old.z+dz));
                if(!next||!next.ports.has(opposite(exit))){this.derail(train);break;}
                train.progress-=1;train.x=next.x;train.z=next.z;train.entry=opposite(exit);track=next;
                const route=this.exits(next,train.entry)[0];
                if(route===undefined){
                    // A disconnected end is a buffer stop, not a teleport.
                    train.exit=train.entry;train.progress=0;train.wait=1;
                    train.exit=opposite(train.entry);
                    this.derail(train);break;
                }
                train.exit=route;
                if(track.kind==='station')train.wait=1.5;
            }
            if(train.guided)Object.assign(train,this.sample(track,train.entry,train.exit,train.progress));
        }
        for(const train of this.active)if(train.guided)this.followTrailer(train);
        // Kinematic guidance is authoritative on the rails: stop before the
        // target poses intersect rather than asking two kinematic bodies to
        // push each other. Rollback propagates through a queue of vehicles.
        for(let pass=0;pass<this.active.length;pass++){
            let changed=false;
            for(let i=0;i<this.active.length;i++)for(let j=i+1;j<this.active.length;j++){
                const a=this.active[i]!,b=this.active[j]!;
                if(!a.guided||!b.guided||!bodiesOverlap(a,b))continue;
                if(!this.stopped.has(a.id)){this.stop(a);changed=true;}
                if(!this.stopped.has(b.id)){this.stop(b);changed=true;}
            }
            if(!changed)break;
        }
        for(const train of this.active)if(train.guided&&!train.blocked)train.trail?.append(train.position);
    }
    reset(): void {
        this.tracks.clear();this.heights.clear();this.trains.clear();this.starts.clear();this.active.length=0;this.stopped.clear();this.running=true;
        for(let z=-this.radius;z<this.radius;z++)for(let x=-this.radius;x<this.radius;x++)this.dirty.add(cellKey(x,z));
        // Loop laid in travel order, so adjacent sides never steal a port.
        for(let x=-3;x<=3;x++)this.placeTrack(x,-3);
        for(let z=-2;z<=3;z++)this.placeTrack(3,z);
        for(let x=2;x>=-3;x--)this.placeTrack(x,3);
        for(let z=2;z>=-2;z--)this.placeTrack(-3,z);
        this.placeTrack(0,3,'station');
    }
}

/** Fill every crossed tile even if pointer events skip multiple cells. */
export function tileStroke(from: Cell,to: Cell): Cell[] {
    const cells:Cell[]=[];let [x,z]=from;
    // Manhattan steps guarantee that every new rail has a cardinal connector.
    while(x!==to[0]||z!==to[1]){
        if(Math.abs(to[0]-x)>=Math.abs(to[1]-z)&&x!==to[0])x+=Math.sign(to[0]-x);
        else z+=Math.sign(to[1]-z);
        cells.push([x,z]);
    }
    return cells;
}
