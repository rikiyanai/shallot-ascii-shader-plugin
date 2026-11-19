import {nativeExpressionKey} from '../text/native-expression.ts';
import {nativeSpacePlan} from '../text/native-whitespace.ts';

/** Completed offline native drawings. Runtime selects a sampled view/size;
 * it neither fits glyphs nor reconstructs passages from a source image. */
export interface NativePhysicalWordWitness {
  readonly row:number;readonly x:number;readonly text:string;readonly advance:number;readonly feature:string;
  readonly qualified:boolean;readonly qualification:string;
  readonly failureCauses?:readonly string[];
  readonly admission:{readonly namedSelectedSource:string|null;readonly wholeMaskSubjectSurface:boolean;readonly bothOriginalPorts:boolean};
  readonly ports:readonly {readonly sourcePathId:string;readonly sourceEdge?:string;readonly participants:readonly unknown[];
    /** v2: verified original route order, not an ink-endpoint alias. */
    readonly sourceOrderId?:string;readonly sourceVersion?:string;
    readonly sourceInput?:string;readonly sourceInputSha256?:string;
    /** Explicit authored orientation operands; per-interval originals remain authoritative. */
    readonly authoredComposition?:Readonly<Record<string,unknown>>;
    readonly neighbourWholeMaskSubjectSurface?:boolean;
    readonly intervals?:readonly Readonly<Record<string,unknown>>[];
    readonly orderAuthority?:string;readonly portMeaning?:string;
    /** Legacy pair exists ONLY when every authoritative interval is on the same edge. */
    readonly range?:readonly number[];readonly entry:Readonly<Record<string,unknown>>;readonly exit:Readonly<Record<string,unknown>>;
    readonly originalEntry:unknown;readonly originalExit:unknown}[];
  readonly ink:readonly Readonly<Record<string,unknown>>[];
  /** Actual first/last native mask pixel ray hits, NOT source continuation
   * incidence. May be null/off-subject and does not imply qualification. */
  readonly physicalInkPorts?:{readonly entry:Readonly<Record<string,unknown>>|null;readonly exit:Readonly<Record<string,unknown>>|null};
}
export interface NativePhysicalFrame {
  readonly schema:'studio.physical-frame.v1';readonly subject:string;
  readonly camera:{readonly eye:readonly number[];readonly right:readonly number[];readonly up:readonly number[];readonly forward:readonly number[];readonly focal:number;readonly orthographic:boolean};
  readonly sun?:Readonly<Record<string,unknown>>;
  readonly nativeChart:Readonly<Record<string,unknown>>;
  readonly provenance:Readonly<Record<string,unknown>>;
  readonly summary:Readonly<Record<string,unknown>>;
  readonly limitations:readonly string[];
}
export interface CompiledNativeWord {
  readonly x:number; readonly row:number; readonly text:string; readonly advance:number;
  readonly feature:string; readonly role:string;
  /** Source-selected contact anchors retained by the offline compiler. */
  readonly anchorRoleIndices?:readonly number[];
  /** Semantic references resolve to this sample's indices at residency upload. */
  readonly anchorIds?:readonly string[];
  readonly spanIds?:readonly string[];
  /** Persistent span expression identity, distinct from exact contact ports. */
  readonly spanTrack?:string;
  readonly atomicConstruction?:string;
  /** SAME selected Studio word and actual physical ink/ORIGINAL ports. False
   * remains false; a runtime adapter must not promote missing correspondence. */
  readonly physicalWitness?:NativePhysicalWordWitness;
  /** Admitted source-role articulation on this word's shadow-side arc. */
  readonly shadow?:readonly {readonly text:string;readonly x:number;readonly advance:number;readonly side:0|1;readonly source:string;readonly minimumOrigin?:number}[];
}
export interface CompiledNativeAnchor {
  readonly index:number;readonly role:string;readonly point:readonly [number,number];
  readonly id?:string;readonly feature?:string;
  readonly identityScope?:'source'|'feature-role'|'sample';
  readonly sourceRoleEntry?:string;readonly semanticPhase?:number;
}
/** A source-selected multi-anchor word OR explicit authored membership between
 * semantic endpoints. Never inferred from proximity/row order. */
export interface CompiledNativeSpan {
  readonly id:string;readonly feature:string;readonly anchorIds:readonly string[];
  /** Ordered whole words may differ in count between sampled articulations. */
  readonly wordIndices?:readonly number[];
  readonly authorship?:'source-manifest';readonly phase?:'not-recorded';
}
export interface CompiledNativeSample {
  readonly view:number; readonly radius:number;
  readonly anchor:readonly [number,number];
  readonly extent:readonly [number,number];
  readonly words:readonly CompiledNativeWord[];
  readonly anchors?:readonly CompiledNativeAnchor[];
  readonly spans?:readonly CompiledNativeSpan[];

  readonly sourceDoc?:Readonly<Record<string,unknown>>;
  readonly source:string;
  /** Physical samples retain the complete caller camera, never frame%96. */
  readonly physicalFrame?:NativePhysicalFrame;
}
export interface CompiledNativeTable {
  readonly schema:'trueff.native_table.v1';
  readonly fontPx:16; readonly rowPitch:17;
  readonly samples:readonly CompiledNativeSample[];
  /** Direct index [view][size], compiled rather than searched during display. */
  readonly views:readonly number[];
  readonly radii:readonly number[];
  readonly viewLookup:readonly number[];
  readonly radiusThresholds:readonly number[];
  readonly tonePassages?:readonly (readonly string[])[];
  readonly index:readonly (readonly number[])[];
  /** Sphere-envelope tables are sized and centred by the projected silhouette
   * of the bound sphere (vertical tangent extent), not focal*radius/depth.
   * Absent: the original radius estimate and LOD thresholds are unchanged. */
  readonly selection?:'projected-silhouette-height';
  /** Physical words are valid only for their exact saved caller request.
   * Their legacy azimuth/radius lookup arrays are empty, not an applicability predicate. */
  readonly sampleSelection?:{readonly kind:'exact-saved-request';readonly frames:readonly {
    readonly sample:number;readonly viewKey:string;readonly savedRequestSHA256:string;
  }[]};
  readonly correspondence?:{
    readonly scope:'feature-semantic-anchors-and-selected-word-spans'|'feature-semantic-anchors-and-source-spans';
    readonly fullSourceDomain:'not-recorded';
    readonly anchors:readonly {readonly id:string;readonly feature:string;readonly samples:readonly {readonly sample:number;readonly index:number}[]}[];
    readonly spans:readonly {readonly id:string;readonly feature:string;readonly anchorIds?:readonly string[];readonly samples:readonly number[]}[];
    readonly limitations:readonly string[];
  };
  readonly provenance:{readonly compiler:string;readonly sourceSha256:Readonly<Record<string,string>>;readonly maxWordShift:number};
}

/** Small foreground circles express shadow through contour weight. The
 * aa64 lower comma pair hands to aa65's enclosing brace, not an interior fill.
 * A left-shadow row retains its leading X and paired interior gap. */
export function smallCircleShadowArcs(table:CompiledNativeTable):CompiledNativeTable {
  return {...table,samples:table.samples.map(sample=>{
    if(!sample.source.endsWith('#sh-lib64-circle-4row'))return sample;
    const row=Math.max(...sample.words.map(w=>w.row))-1;
    const flanks=sample.words.filter(w=>w.row===row).sort((a,b)=>a.x-b.x);
    const left=flanks[0],right=flanks.at(-1);if(!left||!right||left===right)return sample;
    const minimumOrigin=11-left.x;
    return {...sample,words:sample.words.map(word=>word===left?{...word,shadow:[{text:'{',x:word.x,advance:4,side:1 as const,minimumOrigin,source:'role_lookup.v1.json#sh-lib65-circle-3row-brace/r2'}]}:
      word===right?{...word,shadow:[
        {text:'}',x:word.x,advance:4,side:0 as const,source:'role_lookup.v1.json#sh-lib65-circle-3row-brace/r2'},
        {text:word.text,x:word.x-2,advance:word.advance,side:1 as const,minimumOrigin,source:sample.source+'/paired-lower-flank'},
      ]}:word)};
  })};
}

/** Explicit sphere-envelope repair, never a global punctuation rewrite.
 * Lower flanks continue their own side before the paired base return. */
export function orientSphereFlanks(table:CompiledNativeTable):CompiledNativeTable {
  return {...table,samples:table.samples.map(sample=>({...sample,words:sample.words.map(word=>{
    if(word.role==='tone'||word.row*17+8<sample.anchor[1]||!['（','）'].includes(word.text))return word;
    return {...word,text:word.x+word.advance/2<sample.anchor[0]?'（':'）'};
  })}))};
}

/** Residency upload: bind finished native samples to explicit source subjects.
 * The resulting addresses live in the existing feature input buffer. */
export function appendNativeTables(words:number[],subjects:readonly {nativeDrawing?:CompiledNativeTable;nativeTone?:'baked'|'regional';center:readonly number[];radius:number}[],
    paths:Uint32Array,pathCount:number,glyph:(ch:string)=>readonly [number,number],compactTone:readonly (readonly string[])[]=[],extraTables:readonly CompiledNativeTable[]=[]) {
  const roots=new Map<number,number[]>(),bindings:{instance:number;path:number;sources:number[];table:CompiledNativeTable}[]=[];
  // Loop 336 (setScene stage timing: feature routes ~110 ms per Y8 upload): the
  // stroke paths of each native subject and variant, grouped in ONE pass in path
  // order, instead of scanning every packet path for every native subject
  // (subjects x paths). Same source lists, same order.
  const byInstance=new Map<number,[number[],number[]]>();
  for(let p=0;p<pathCount;p++){
    if(paths[p*8+7]!==0||((paths[p*8+3]!>>>16)&15)!==0)continue;
    const instance=paths[p*8+2]!;if(!subjects[instance]?.nativeDrawing)continue;
    let group=byInstance.get(instance);if(!group){group=[[],[]];byInstance.set(instance,group);}
    group[(paths[p*8+3]!>>>20)&1]!.push(p);
  }
  subjects.forEach((s,instance)=>{if(!s.nativeDrawing)return;
    for(let variant=0;variant<2;variant++){
      const sources=byInstance.get(instance)?.[variant]??[];
      if(sources.length)bindings.push({instance,path:sources[0]!,sources,table:s.nativeDrawing});
    }
  });
  const F=(n:number)=>new Uint32Array(new Float32Array([n]).buffer)[0]!;
  const reserve=(n:number)=>{const at=words.length;for(let i=0;i<n;i++)words.push(0);return at;};
  const bindingAt=reserve(bindings.length*8),tables=new Map<CompiledNativeTable,number>();
  // Loop 337: one table serializer (first use, same order as before); extraTables (dynamic
  // slot phrases) are appended after the bindings so a slot can switch table in place.
  const serialize=(table:CompiledNativeTable):number=>{
    if(table.sampleSelection?.kind==='exact-saved-request'||table.samples.some(s=>s.physicalFrame))throw Error('Physical native table needs its exact saved-request/camera consumer; azimuth/size residency cannot place it');
    let at=tables.get(table);
    if(at===undefined){
      at=reserve(8);tables.set(table,at);
      const lookup=words.length;words.push(...table.viewLookup);
      // A complete catalog circle cannot be scaled down to the midpoint
      // between row footprints: its cap/flanks/return are intact native rows.
      // Admit the next size only once that whole footprint fits. Other kinds
      // of drawing keep their separately authored LOD thresholds unchanged.
      const wholeCircles=table.samples.every(sample=>sample.words.length>0&&sample.words.every(word=>word.role==='whole_circle'));
      const thresholds=words.length;words.push(...(wholeCircles?table.radii.slice(1):table.radiusThresholds).map(F));
      const index=words.length;words.push(...table.index.flat());
      const samples=reserve(table.samples.length*8);
      table.samples.forEach((sample,sampleIndex)=>{
        const namedAnchors=new Map((sample.anchors??[]).filter(a=>a.id).map(a=>[a.id!,a]));
        const namedSpans=new Map((sample.spans??[]).map(s=>[s.id,s]));
        const first=words.length;let count=0;const shadows:{at:number;word:CompiledNativeWord}[]=[];
        for(const word of sample.words){
          // Resolve IDs in the selected sample, never reuse another sample's
          // ordinal. The existing GPU anchorRoleIndices mask is the consumer.
          const ids=[...(word.anchorIds??[])];
          for(const id of word.spanIds??[]){
            const span=namedSpans.get(id);
            // A long authored span does NOT make every intervening word touch
            // both endpoint ports. Only actual source-selected contacts join
            // the contact mask; span identity uses the expression lane below.
            if(span&&span.authorship!=='source-manifest')ids.push(...span.anchorIds.filter(anchorId=>namedAnchors.get(anchorId)?.feature===span.feature));
          }
          // New ID/span references are validated by the offline compiler. Keep
          // the existing bounded mask/publication behavior for legacy inputs.
          const indices=ids.length?[...new Set(ids)].flatMap(id=>{
            const anchor=namedAnchors.get(id);return anchor?[anchor.index]:[];
          }):word.anchorRoleIndices??[];
          const anchorMask=indices.reduce((mask,index)=>index>=0&&index<32?mask|(1<<index):mask,0)>>>0;
          const chars=[...word.text];let pen=word.x;
          for(let start=0;start<chars.length;start+=6){
            const part=chars.slice(start,start+6),packed=[0,0];let advance=0;
            part.forEach((ch,j)=>{const [code,width]=glyph(ch);packed[Math.floor(j/3)]!|=code<<((j%3)*10);advance+=width;});
            if(word.shadow&&start===0)shadows.push({at:words.length,word});
            words.push(pen>>>0,word.row>>>0,part.length,packed[0]!>>>0,packed[1]!>>>0,advance,Number(word.role==='tone'),nativeExpressionKey(word.spanTrack??word.feature),anchorMask);
            pen+=advance;count++;
          }
        }
        for(const {at,word}of shadows){
          words[at+6]=(words.length*4+2)>>>0;
          words.push(word.shadow!.length);
          for(const shadow of word.shadow!){
            const chars=[...shadow.text],packed=[0,0];chars.forEach((ch,i)=>packed[Math.floor(i/3)]!|=glyph(ch)[0]<<((i%3)*10));
            words.push(shadow.x,chars.length,packed[0]!>>>0,packed[1]!>>>0,shadow.advance,shadow.side,(shadow.minimumOrigin??-2147483648)>>>0);
          }
        }
        const anchorAt=words.length;
        for(const anchor of sample.anchors??[])words.push(anchor.index,F(anchor.point[0]),F(anchor.point[1]),nativeExpressionKey(anchor.role));
        // Source-bound tone positions are admitted ONCE against this exact
        // silhouette's interior scalar intervals. Display chooses role/side,
        // never fits or clips a floating passage against the physical mesh.
        const toneAt=reserve(6),lastRow=Math.max(...sample.words.map(w=>w.row));
        for(let reverse=0;reverse<2;reverse++)for(let level=0;level<3;level++){
          const descriptor=words.length;words[toneAt+reverse*3+level]=descriptor;words.push(0);let toneCount=0;
          // A source small-circle shadow arc owns its shading: do not compete
          // with it using a squeezed interior field in the two available rows.
          const rows=shadows.length?[]:table.tonePassages?.[level]??compactTone[level]??[],start=Math.max(1,Math.floor((lastRow+1-Math.min(3,lastRow-1))/2));
          rows.forEach((source,r)=>{
            const row=start+r;if(row>=lastRow)return;
            const sides=sample.words.filter(w=>w.row===row&&w.role!=='tone').sort((a,b)=>a.x-b.x);
            if(sides.length<2)return;
            const left=sides[0]!.x+sides[0]!.advance,right=sides.at(-1)!.x;
            const chars=reverse?[...source].reverse():[...source],advance=chars.reduce((n,ch)=>n+glyph(ch)[1],0);
            const wanted=reverse?left+5:right-advance-5;
            let x:number|undefined,best=Infinity;
            for(let pen=left+5;pen+advance<=right-5;pen++){
              if(!nativeSpacePlan(pen-left)||!nativeSpacePlan(right-pen-advance))continue;
              const cost=Math.abs(pen-wanted);if(cost<best){x=pen;best=cost;}
            }
            if(x===undefined)return;
            for(let at=0;at<chars.length;at+=6){
              const part=chars.slice(at,at+6),packed=[0,0];let width=0;
              part.forEach((ch,i)=>{const[code,adv]=glyph(ch);packed[Math.floor(i/3)]!|=code<<((i%3)*10);width+=adv;});
              words.push(x,row,part.length,packed[0]!>>>0,packed[1]!>>>0,width);x+=width;toneCount++;
            }
          });words[descriptor]=toneCount;
        }
        words.splice(samples+sampleIndex*8,8,first,count,F(sample.anchor[0]),F(sample.anchor[1]),F(sample.radius),anchorAt,(sample.anchors??[]).length,toneAt);
      });
      words.splice(at,8,table.views.length,table.radii.length,lookup,thresholds,index,samples,Number(table.selection==='projected-silhouette-height'),0);
    }
    return at;
  };
  const tableWords=new Map<number,number[]>();
  bindings.forEach((binding,i)=>{
    const at=serialize(binding.table);
    const subject=subjects[binding.instance]!,b=bindingAt+i*8;
    const pointers=tableWords.get(binding.instance)??[];pointers.push(b+2);tableWords.set(binding.instance,pointers);
    const visibility=words.length;words.push(binding.sources.length,...binding.sources);
    words.splice(b,8,(binding.instance|(subject.nativeTone==='regional'?0x80000000:0))>>>0,binding.path,at,visibility,...subject.center.map(F),F(subject.radius));
    const offsets=roots.get(binding.instance)??[];offsets.push(b+4);roots.set(binding.instance,offsets);
  });
  for(const table of extraTables)serialize(table);
  return{count:bindings.length,at:bindingAt,roots,tables,tableWords};
}
