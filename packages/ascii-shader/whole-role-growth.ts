import roles from '../../assets/grammar/aahub-role-lookup-3493185.v1.json' with {type:'json'};
import supplement from './grammar/role-rounds-supplement.json' with {type:'json'};
import {nativeSpacePlan} from './text/native-whitespace.ts';
import {relationPoint,relationParameter,type PosedRelation} from './posed-graph.ts';
import type {FeatureWalk} from './feature-graph.ts';
import type {NativePattern} from './native-layout.ts';
import type {NativeGlyph,NativeOccurrence,NativeContact,NativePaper} from './native-page.ts';

interface Entry {id:string;native:{rows:string[];row_numbers:number[];row_pen_x:number[]};source:{src:string}}
interface Word {chars:{text:string;cp:number}[]}
interface RowSource {entry:Entry;sourceRow:number;words:Word[];gaps:number[]}
interface RoleRow {id:string;source:RowSource;part:'cap'|'flank'|'return';upright:boolean}
interface Token {text:string;x:number;row:number;id:string;word:number;cp:number;inserted:boolean}
interface RowCandidate {role:RoleRow;row:number;tokens:Token[];left:Token[];right:Token[];
  cost:number;leftPen:number;rightPen:number;key:string;gapRects:readonly [number,number][]}
type Point=readonly [number,number];
interface Handoff {a:Token;b:Token;p:Point;q:Point;squared:number}
export interface GrowthState {rowCount:number;rows:ReadonlyMap<string,string>}
export interface GrowthTerminal {readonly row:number;readonly left:number;readonly floor:number;
  readonly rail?:{readonly edge:PosedRelation;readonly start:number;readonly end:number;readonly reverse:boolean};}
export interface GrowthWitness {occurrence:string;source:string;entry:string;sourceRow:number;sourceCP:number;
  role:string;event:'retained-source'|'inserted-whole-unit';rule?:'provisional-width-growth'}
export interface GrowthRegionSlot {id:string;row:number;left:number;right:number;owner:string;
  sourceRegions:readonly string[];leftRole:string;rightRole:string}
export type GrowthResult = {ok:true;pattern:NativePattern;state:GrowthState;witnesses:GrowthWitness[];
  regionSlots:GrowthRegionSlot[];handoffs:readonly {from:string;to:string;kind:'ink-contact'|'implied-paper';
    fromAnchor:Point;toAnchor:Point;rule:'provisional-source-role-growth'}[]}
  |{ok:false;id:string;obligation:string};

/** Whole source-role growth. The external seam takes a COMPLETE walk; intervals
 * are deliberately never inspected. Visibility belongs AFTER this producer.
 * There is no scene name, circle fit, font enumeration, or visible-row index.
 * Cross-entry assembly/width growth are PROVISIONAL rules from roundcomp.program,
 * not additional observed source annotations. */
export class WholeRoleGrowth {
  private readonly font:Map<string,NativeGlyph>;
  private readonly entries=new Map<string,Entry>();
  private readonly parsed=new Map<string,RowSource>();
  private readonly interfaces=new Map<string,{p:Point;q:Point;squared:number}|null>();
  private readonly cap:RowSource[];private readonly ret:RowSource[];
  private readonly brace:RowSource;private readonly upright:RowSource;
  private readonly paperLimitSquared:number;
  constructor(font:readonly NativeGlyph[]) {
    this.font=new Map(font.map(g=>[g.text,g]));
    for(const entry of [...roles.entries,...supplement.entries] as unknown as Entry[])this.entries.set(entry.id,entry);
    this.cap=this.rows('sh-lib57-cap-shoulders');this.ret=this.rows('sh-lib57-return-base');
    this.brace=this.rows('sh-lib65-circle-3row-brace')[1]!;
    this.upright=this.rows('sh-lib57-flanks')[1]!;
    // The actual lib57 l→､ lower shoulder has15.81px of implied paper.
    // NativeProgram's12px storage limit is not a universal contour law.
    // Compile this source-function allowance from its exact source rows.
    const e=this.entries.get('sh-lib57-flanks')!;
    const reference=[2,3].map(r=>{
      let x=e.native.row_pen_x[r]!;const out:{text:string;x:number;y:number}[]=[];
      for(const text of e.native.rows[r]!.replace(/\\u([0-9a-fA-F]{4})/g,(_,cp:string)=>String.fromCharCode(parseInt(cp,16)))) {const g=this.font.get(text)!;if(g.points.length)out.push({text,x,y:r*17});x+=g.advance;}
      return out;
    });
    this.paperLimitSquared=Math.max(...[0,1].map(side=>{
      const a=side?reference[0]!.at(-1)!:reference[0]![0]!,b=side?reference[1]!.at(-1)!:reference[1]![0]!;
      let squared=Infinity;
      for(const p of this.font.get(a.text)!.points)for(const q of this.font.get(b.text)!.points)
        squared=Math.min(squared,(a.x+p[0]!-b.x-q[0]!)**2+(a.y+p[1]!-b.y-q[1]!)**2);
      return squared;
    }));
  }
  private advance(text:string) {const glyph=this.font.get(text);if(!glyph)throw Error(`Unavailable native scalar ${text}`);return glyph.advance;}
  private wordWidth(word:Word) {return word.chars.reduce((n,c)=>n+this.advance(c.text),0);}
  private rows(id:string):RowSource[] {
    const entry=this.entries.get(id);if(!entry)throw Error(`Missing source role ${id}`);
    return entry.native.rows.map((encoded,index)=>{
      const key=`${id}/${index}`,old=this.parsed.get(key);if(old)return old;
      const text=encoded.replace(/\\u([a-fA-F0-9]{4})/g,(_,cp)=>String.fromCharCode(parseInt(cp,16)));
      const words:Word[]=[],gaps:number[]=[];let cp=0,gap=0,word:Word|undefined;
      for(const char of text) {
        if(char===' '||char==='　') {gap+=this.advance(char);word=undefined;}
        else {if(!word){if(words.length)gaps.push(gap);word={chars:[]};words.push(word);gap=0;}word.chars.push({text:char,cp});}
        cp++;
      }
      const row={entry,sourceRow:entry.native.row_numbers[0]!+index,words,gaps};this.parsed.set(key,row);return row;
    });
  }
  private program(n:number):RoleRow[] {
    const role=(id:string,source:RowSource,part:RoleRow['part'],upright=false):RoleRow=>({id,source,part,upright});
    if(n<=4) {
      const id=n===2?'sh-lib66-circle-2row':n===3?'sh-lib65-circle-3row-brace':'sh-lib64-circle-4row';
      return this.rows(id).map((row,i,rs)=>role(i===0?'cap/crown':i===rs.length-1?'return/base':`flank/small-${i}`,row,i===0?'cap':i===rs.length-1?'return':'flank'));
    }
    const cap=n<7?[0,2]:[0,1,2],ret=n<7?[0,2]:[0,1,2],m=n-cap.length-ret.length;
    // Growth inserts outward from an enduring equator. Existing IDs do not
    // change parity or get renumbered when another row is inserted.
    const middle:{side:'north'|'south'|'equator';radius:number}[]=[{side:'equator',radius:0}];
    for(let k=1;middle.length<m;k++)middle.push({side:k%2?'north':'south',radius:Math.ceil(k/2)});
    middle.sort((a,b)=>(a.side==='north'?-a.radius:a.radius)-(b.side==='north'?-b.radius:b.radius));
    return [...cap.map(i=>role(`cap/${i===0?'crown':i===1?'outer-shoulder':'inner-shoulder'}`,this.cap[i]!,'cap')),
      ...middle.map(p=>role(`flank/${p.side}-${p.radius}`,p.radius%2?this.upright:this.brace,'flank',p.radius%2===1)),
      ...ret.map(i=>role(`return/${i===0?'inner-shoulder':i===1?'outer-shoulder':'base'}`,this.ret[i]!,'return'))];
  }
  private railProgram(n:number):RoleRow[]{
    const terminal:RoleRow={id:'return/rail-base',source:this.rows('sh-lib66-circle-2row')[1]!,part:'return',upright:false};
    if(n<=3)return [...this.program(n).slice(0,-1),terminal];
    const upper=this.program(n+1).filter(row=>row.part!=='return');
    return [...upper,...(n>=6?[{id:'return/outer-shoulder',source:this.ret[1]!,part:'return' as const,upright:false}]:[]),terminal];
  }
  private terminal(role:RoleRow,row:number,origin:number,input:GrowthTerminal):RowCandidate{
    const source=role.source,leftWord=source.words[0]!,rightWord=source.words[1]!,tokens:Token[]=[];
    let pen=input.left;
    const emit=(text:string,cp:number,word:number,inserted=false,unit=0)=>{
      tokens.push({text,cp,word,inserted,row,x:pen,id:`${role.id}/${source.entry.id}/${source.sourceRow}/${inserted?`growth-${cp}-${unit}`:`source-${cp}`}`});pen+=this.advance(text);
    };
    for(const char of leftWord.chars)emit(char.text,char.cp,0);
    const gapStart=pen,gap=source.gaps[0]!;pen+=gap;
    const repeats=rightWord.chars.filter(char=>char.text==='_'),end=rightWord.chars.at(-1)!;
    for(let i=0;i<input.floor;i++){const char=repeats[Math.min(i,repeats.length-1)]!;emit(char.text,char.cp,1,i>=repeats.length,i-repeats.length);}
    emit(end.text,end.cp,1);
    return {role,row,tokens,left:tokens.filter(token=>token.word===0),right:tokens.filter(token=>token.word===1),cost:0,
      leftPen:input.left,rightPen:pen,key:`rail/${input.left-origin}/${input.floor}`,gapRects:[[gapStart,gap]]};
  }
  /** Native PAPER slot alternatives. They alter constructor parameters before
   * literal pens exist; no placed word is subsequently nudged. Fixed tiny bound. */
  private slots(wanted:number,min:number,leading:boolean,limit=3):number[] {
    const out:number[]=[];
    // Test reachability, do not serialize whitespace here. Near-plane source
    // spans can be enormous; only NativePage emits their visible scalar tape.
    for(let d=0;d<=16&&out.length<limit;d++)for(const value of d?[Math.round(wanted)-d,Math.round(wanted)+d]:[Math.round(wanted)])
      if(value>=min&&!out.includes(value)&&nativeSpacePlan(value,0,leading))out.push(value);
    return out;
  }
  private repeat(row:RoleRow):{word:number;cp:number;text:string}|undefined {
    if(row.part==='flank')return;
    // A rule names a slot IN this source row; it does not choose a glyph by
    // orientation. Prefer an interior source word when one exists.
    const order=row.source.words.length>2?[1,...row.source.words.keys()]:[...row.source.words.keys()];
    for(const unit of ['─','⌒','＿','―','ｰ','-','_'])for(const wi of order) {
      const found=row.source.words[wi]!.chars.find(c=>c.text===unit);
      if(found)return {word:wi,cp:found.cp,text:unit};
    }
  }
  private candidates(role:RoleRow,row:number,left:number,right:number,origin:number,previous:string|undefined):RowCandidate[] {
    const source=role.source,repeat=this.repeat(role),baseWords=source.words.reduce((n,w)=>n+this.wordWidth(w),0);
    const baseSpan=baseWords+source.gaps.reduce((n,g)=>n+g,0),unit=repeat?this.advance(repeat.text):0;
    // -1 removes the one named repeat-slot unit BETWEEN retained anchors;
    // it never removes a shoulder/end glyph or an independently placed word.
    const grow=repeat?Math.max(-1,Math.round((right-left-baseSpan)/unit)):0;
    const counts=repeat?[...new Set([Math.max(-1,grow-1),grow,grow+1])]:[0];
    const main=source.gaps.length?source.gaps.indexOf(Math.max(...source.gaps)):-1;
    const candidates:RowCandidate[]=[];
    for(const leftSlot of this.slots(left-origin,0,origin===0))for(const count of counts) {
      const scale=Math.min(1,(right-left)/Math.max(1,baseSpan));
      const fixedGaps=source.gaps.map(g=>role.part==='flank'?g:(this.slots(Math.max(5,g*scale),5,false,1)[0]??g));
      const fixedGap=fixedGaps.reduce((n,g,i)=>n+(i===main?0:g),0);
      const wordsWidth=baseWords+count*unit;
      const variable=main<0?[0]:this.slots(right-(origin+leftSlot)-wordsWidth-fixedGap,
        role.part==='flank'?11:5,false);
      for(const gap of variable) {
        const gaps=[...fixedGaps];if(main>=0)gaps[main]=gap;
        let pen=origin+leftSlot;const tokens:Token[]=[],gapRects:[number,number][]=[];
        for(const [wi,word]of source.words.entries()) {
          if(wi) {gapRects.push([pen,gaps[wi-1]!]);pen+=gaps[wi-1]!;}
          for(const char of word.chars) {
            if(count===-1&&repeat?.word===wi&&repeat.cp===char.cp)continue;
            tokens.push({text:char.text,x:pen,row,id:`${role.id}/${source.entry.id}/${source.sourceRow}/source-${char.cp}`,word:wi,cp:char.cp,inserted:false});pen+=this.advance(char.text);
            if(repeat?.word===wi&&repeat.cp===char.cp)for(let k=0;k<count;k++) {
              tokens.push({text:repeat.text,x:pen,row,id:`${role.id}/${source.entry.id}/${source.sourceRow}/growth-${repeat.cp}-${k}`,word:wi,cp:repeat.cp,inserted:true});pen+=unit;
            }
          }
        }
        // A whole source shoulder word can have different incoming/outgoing
        // participants (e.g.｀ and、). Keep both; inserted cap middle units
        // cannot donate a missing side interface.
        const first=source.words.length===1?[tokens[0]!]:tokens.filter(t=>t.word===0&&!t.inserted);
        const last=source.words.length===1?[tokens.at(-1)!]:tokens.filter(t=>t.word===source.words.length-1&&!t.inserted);
        const key=`${leftSlot}/${count}/${gap}`,cost=(origin+leftSlot-left)**2+(pen-right)**2+(previous===key?0:.01);
        if(Math.abs(origin+leftSlot-left)<=17&&Math.abs(pen-right)<=17)
          candidates.push({role,row,tokens,left:first,right:last,cost,leftPen:origin+leftSlot,rightPen:pen,key,gapRects});
      }
    }
    // Fixed maximum is a finite grammar-state budget, not a font/page search.
    return candidates.sort((a,b)=>a.cost-b.cost||a.key.localeCompare(b.key)).slice(0,8);
  }
  private handoff(before:Token[],after:Token[]):Handoff|undefined {
    let best:Handoff|undefined;
    for(const a of before)for(const b of after) {
      const dx=b.x-a.x,dy=(b.row-a.row)*17,key=`${a.text}/${b.text}/${dx}/${dy}`;
      let cached=this.interfaces.get(key);
      if(cached===undefined) {
        let local:{p:Point;q:Point;squared:number}|null=null;
        for(const p of this.font.get(a.text)!.points)for(const q of this.font.get(b.text)!.points) {
          const squared=(p[0]!-q[0]!-dx)**2+(p[1]!-q[1]!-dy)**2;
          if(squared<=this.paperLimitSquared&&(!local||squared<local.squared))local={p:[p[0]!+.5,p[1]!+.5],q:[q[0]!+.5,q[1]!+.5],squared};
        }
        cached=local;this.interfaces.set(key,cached);
      }
      if(cached&&(!best||cached.squared<best.squared))best={a,b,
        p:[a.x+cached.p[0],a.row*17+cached.p[1]],q:[b.x+cached.q[0],b.row*17+cached.q[1]],squared:cached.squared};
    }
    // Fixed literal interfaces are checked for contact/paper compatibility.
    // This distance NEVER selects a glyph or scores its fit to geometry.
    return best&&best.squared<=this.paperLimitSquared?best:undefined;
  }
  grow(walk:FeatureWalk,previous?:GrowthState,terminal?:GrowthTerminal):GrowthResult {
    const fail=(obligation:string):GrowthResult=>({ok:false,id:walk.id,obligation});
    if(!walk.closed||walk.steps.length<3)return fail('whole rounded growth requires a closed full feature walk');
    const points=walk.steps.map(step=>relationPoint(step.relation,step.reverse?1:0));
    if(points.some(p=>!p.every(Number.isFinite)))return fail('full source envelope crosses the projective eye plane');
    const lo=Math.min(...points.map(p=>p[1])),hi=Math.max(...points.map(p=>p[1])),xmin=Math.min(...points.map(p=>p[0]));
    let n=Math.max(2,Math.round((hi-lo)/17)+1);
    if(previous&&!terminal&&Math.abs((hi-lo)-(previous.rowCount-1)*17)<=10)n=previous.rowCount;
    const initial=this.program(n),initialInk=Math.min(...initial[0]!.source.words.flatMap(w=>w.chars.flatMap(c=>this.font.get(c.text)!.points.map(p=>p[1]!))));
    const anchoredRow=terminal?Math.round((lo-initialInk)/17):undefined;
    if(terminal){n=terminal.row-anchoredRow!+1;if(n<2)return fail('shared return lies above the complete cap; a different whole function is required');}
    const program=terminal?this.railProgram(n):initial;
    if(program.some((p,i)=>p.upright&&program[i-1]?.upright))throw Error('UPRIGHT_RUN=1 violated by the role program');
    const firstInk=Math.min(...program[0]!.source.words.flatMap(w=>w.chars.flatMap(c=>this.font.get(c.text)!.points.map(p=>p[1]!))));
    const lastInk=Math.max(...program.at(-1)!.source.words.flatMap(w=>w.chars.flatMap(c=>this.font.get(c.text)!.points.map(p=>p[1]!))));
    const rowOrigin=anchoredRow??Math.round(((lo+hi)-(firstInk+(n-1)*17+lastInk))/34),origin=11*Math.floor(xmin/11);
    const layers:RowCandidate[][]=[];
    for(const [row,role]of program.entries()) {
      if(terminal&&row===program.length-1){layers.push([this.terminal(role,row,origin,terminal)]);continue;}
      // Rows partition the COMPLETE source-height parameter, rather than a
      // visible screen-row enumeration. The native row lattice remains17px;
      // only the source support interval assigned to each role is quantized.
      const bandLo=lo+(hi-lo)*row/n,bandHi=lo+(hi-lo)*(row+1)/n;
      const y=Math.max(lo+.001,Math.min(hi-.001,(bandLo+bandHi)/2));
      const cuts:number[]=[];
      for(let i=0;i<points.length;i++) {
        const a=points[i]!,b=points[(i+1)%points.length]!;
        if((a[1]<=y&&b[1]>y)||(b[1]<=y&&a[1]>y))cuts.push(a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]));
      }
      cuts.sort((a,b)=>a-b);
      if(cuts.length!==2)return fail('this full row has multiple interiors; a branched/opening native function is required');
      // Full GEOMETRIC band extent, including hidden sides. A crown occupies
      // a native row band, not the zero-width mathematical apex at y=lo.
      const bandX:number[]=[];
      for(let i=0;i<points.length;i++) {
        const a=points[i]!,b=points[(i+1)%points.length]!;
        if(a[1]>=bandLo&&a[1]<=bandHi)bandX.push(a[0]);
        for(const boundary of [bandLo,bandHi])if((a[1]<=boundary&&b[1]>boundary)||(b[1]<=boundary&&a[1]>boundary))
          bandX.push(a[0]+(b[0]-a[0])*(boundary-a[1])/(b[1]-a[1]));
      }
      const left=bandX.length?Math.min(...bandX):cuts[0]!,right=bandX.length?Math.max(...bandX):cuts[1]!;
      const candidates=this.candidates(role,row,left,right,origin,previous?.rows.get(role.id));
      if(!candidates.length)return fail(`${role.id}: no whole repeat/paper counts realize the source extent within one native glyph`);
      layers.push(candidates);
    }
    // A bounded whole-FORM margin automaton. Adjacent role rows select their
    // paper/repeat counts together. No image objective; no per-piece selector.
    const costs=layers.map(layer=>layer.map(()=>Infinity)),prior=layers.map(layer=>layer.map(()=>-1));
    costs[0]=layers[0]!.map(c=>c.cost);
    for(let r=1;r<layers.length;r++)for(const [j,next]of layers[r]!.entries())for(const [i,before]of layers[r-1]!.entries()) {
      if(!Number.isFinite(costs[r-1]![i]!))continue;
      if(!this.handoff(before.left,next.left)||!this.handoff(before.right,next.right))continue;
      const cost=costs[r-1]![i]!+next.cost;
      if(cost<costs[r]![j]!) {costs[r]![j]=cost;prior[r]![j]=i;}
    }
    let at=costs.at(-1)!.indexOf(Math.min(...costs.at(-1)!));
    if(!Number.isFinite(costs.at(-1)![at]!))return fail('native role handoffs cannot retain both sides; a different whole construction function is required');
    const chosen:RowCandidate[]=new Array(layers.length);
    for(let r=layers.length-1;r>=0;r--){chosen[r]=layers[r]![at]!;at=prior[r]![at]!;}
    const occurrences:NativeOccurrence[]=[],witnesses:GrowthWitness[]=[],paper:NativePaper[]=[],regionSlots:GrowthRegionSlot[]=[];
    const prefix=(id:string)=>`${walk.id}/${id}`;
    // Every flank belongs to this SAME complete source walk. Rebuilding
    // its full region union per row made large Y8 contours allocate
    // rows × source-edges provenance records before publishing any ink.
    let sourceRegions:readonly string[]|undefined;
    for(const row of chosen) {
      for(const t of row.tokens) {
        const glyph=this.font.get(t.text)!;
        occurrences.push({id:prefix(t.id),text:t.text,x:t.x-origin,row:t.row,
          contributions:[{source:walk.id,kind:'path',pixels:glyph.points.map((_,i)=>i)},
            ...(terminal?.rail&&row.row===chosen.length-1&&t.text==='_'?[{source:terminal.rail.edge.id,kind:'path' as const,pixels:glyph.points.map((_,i)=>i),
              parameter:(()=>{const rail=terminal.rail!,a=Math.max(0,Math.min(1,(t.x+Math.min(...glyph.points.map(p=>p[0]!))-rail.start)/(rail.end-rail.start))),b=Math.max(0,Math.min(1,(t.x+Math.max(...glyph.points.map(p=>p[0]!))-rail.start)/(rail.end-rail.start)));
                const screen=rail.reverse?[1-b,1-a]:[a,b];return [relationParameter(rail.edge,screen[0]!),relationParameter(rail.edge,screen[1]!)] as const;})()}]:[])]});
        witnesses.push({occurrence:prefix(t.id),source:row.role.source.entry.source.src,entry:row.role.source.entry.id,
          sourceRow:row.role.source.sourceRow,sourceCP:t.cp,role:row.role.id,event:t.inserted?'inserted-whole-unit':'retained-source',
          ...(t.inserted?{rule:'provisional-width-growth' as const}:{})});
      }
      if(row.role.part==='flank') {
        const leftEnd=Math.max(...row.left.map(t=>t.x+this.advance(t.text))),rightStart=Math.min(...row.right.map(t=>t.x));
        if(rightStart-leftEnd>6){
          sourceRegions??=[...new Set(walk.steps.flatMap(s=>s.relation.regions.filter((r):r is string=>!!r)))];
          regionSlots.push({id:`${walk.id}/${row.role.id}/receiver`,row:row.row,left:leftEnd-origin+3,right:rightStart-origin-3,
            owner:walk.owner,sourceRegions,leftRole:prefix(row.left.at(-1)!.id),rightRole:prefix(row.right[0]!.id)});
        }
      } else for(const [i,[x,width]]of row.gapRects.entries())if(width>0)
        paper.push({id:`${walk.id}/${row.role.id}/source-paper-${i}`,rect:[x-origin,row.row*17,width,17],sources:[walk.id]});
    }
    const contacts:NativeContact[]=[],handoffs:Extract<GrowthResult,{ok:true}>['handoffs'][number][]=[];
    for(let r=1;r<chosen.length;r++)for(const side of ['left','right'] as const) {
      const h=this.handoff(chosen[r-1]![side],chosen[r]![side])!;
      const from=prefix(h.a.id),to=prefix(h.b.id),a:Point=[h.p[0]-origin,h.p[1]],b:Point=[h.q[0]-origin,h.q[1]];
      handoffs.push({from,to,kind:h.squared<=9?'ink-contact':'implied-paper',fromAnchor:a,toAnchor:b,rule:'provisional-source-role-growth'});
      if(h.squared<=9)contacts.push({id:`${walk.id}/${chosen[r]!.role.id}/${side}/contact`,from,to,fromSource:walk.id,toSource:walk.id,
        fromPort:`${side}/out`,toPort:`${side}/in`,fromAnchor:a,toAnchor:b,relation:'provisional-source-role-growth'});
    }
    return {ok:true,pattern:{id:walk.id,origin:[origin,rowOrigin],occurrences,contacts,paper,requiredSources:[walk.id],
      continuations:handoffs.map(handoff=>({from:handoff.from,to:handoff.to,fromAnchor:handoff.fromAnchor,toAnchor:handoff.toAnchor,distance:Math.sqrt(this.paperLimitSquared)}))},
      state:{rowCount:n,rows:new Map(chosen.map(r=>[r.role.id,r.key]))},witnesses,regionSlots,handoffs};
  }
}
