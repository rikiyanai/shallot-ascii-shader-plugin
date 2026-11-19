import type {NativeGlyph,NativeOccurrence} from './native-page.ts';
import type {NativePattern} from './native-layout.ts';
import type {NativeSupport} from './native-presentation.ts';
import type {DrawingFieldView} from './field-readback.ts';

/** An authored text source has its own lifetime; it is not a triangle mesh or
 * a texture to OCR. This domain is never encoded into the fragment field. */
export const SEMANTIC_TEXT_DOMAIN=0x20000000;
export interface NativeLabel {
 readonly id:string;readonly reference:number;readonly text:string;
 readonly anchor:readonly [number,number,number];readonly pivot:readonly [number,number];
 readonly justify:number;readonly maxEmWidth:number;readonly depthBias:number;
 readonly opacity:number;readonly color:number;readonly camera:string;
}
/** Native paper is white, so neutral authored UI text keeps the existing
 * readable-ink adaptation. A meaningful authored hue, however, is already a
 * foreground choice and must not be converted to a brown/gray luminance.
 * Near-neutral UI colours are included in the adaptation band; this is a
 * colour-space rule, not a scene/effect special case. */
const NEUTRAL_CHROMA_LIMIT=0.1*255;
export function nativeLabelForeground(color:number,opacity:number):number{
 const value=Math.round(color)>>>0,rgb=[value>>>16&255,value>>>8&255,value&255],chroma=Math.max(...rgb)-Math.min(...rgb);
 const luminance=(rgb[0]!*.2126+rgb[1]!*.7152+rgb[2]!*.0722)/255;
 const gain=chroma<=NEUTRAL_CHROMA_LIMIT&&luminance>.25?.25/luminance:1;
 const ink=rgb.map(c=>Math.round(c*gain));
 return ((Math.round(Math.max(0,Math.min(1,opacity))*255)<<24)|ink[2]!<<16|ink[1]!<<8|ink[0]!)>>>0;
}
/** Source strings reflow as WHOLE words in native Saitamaar metrics. Their
 * anchor, content and lifetime come from the same live World as geometry.
 * No original-font quads, pixels or example identifiers select characters. */
export function constructNativeLabels(labels:readonly NativeLabel[],glyphs:readonly NativeGlyph[],fields:readonly DrawingFieldView[]):{
 patterns:NativePattern[];supports:Map<string,NativeSupport>;unresolved:string[];
}{
 const font=new Map(glyphs.map(g=>[g.text,g])),patterns:NativePattern[]=[],supports=new Map<string,NativeSupport>(),unresolved:string[]=[];
 for(const label of labels){
  if([...label.text].some(c=>c!=='\n'&&c!=='\r'&&!font.has(c))){unresolved.push(`${label.id}:authored text contains an unavailable native scalar`);continue;}
  const maxWidth=Number.isFinite(label.maxEmWidth)?Math.max(16,label.maxEmWidth*16):Infinity;
  const lines:{text:string;offset:number}[]=[];let scalar=0;
  for(const explicit of label.text.replace(/\r\n?/g,'\n').split('\n')){
   let line='',lineStart=scalar,width=0;
   for(const word of explicit.match(/\S+\s*|\s+/gu)??['']){
    const advance=[...word].reduce((sum,c)=>sum+font.get(c)!.advance,0);
    if(line&&width+advance>maxWidth){lines.push({text:line,offset:lineStart});lineStart=scalar;line='';width=0;}
    line+=word;width+=advance;scalar+=[...word].length;
   }
   lines.push({text:line,offset:lineStart});scalar++;
  }
  const widths=lines.map(line=>[...line.text].reduce((sum,c)=>sum+font.get(c)!.advance,0)),width=Math.max(...widths),height=lines.length*17;
  const origin=Math.round(label.anchor[0]-width*label.pivot[0]),row=Math.round((label.anchor[1]-height*label.pivot[1])/17);
  // Retained source labels that are wholly hidden stay hidden. Partial
  // clipping/occlusion needs a native word-level construction, not masking
  // font pixels or printing a hidden background label through a solid.
  let hidden=false;
  if(label.depthBias>=0)for(const view of fields){
   if(view.camera!==label.camera)continue;
   const sample=view.field.sample((label.anchor[0]-view.rect.x)/view.rect.width*view.field.width,(label.anchor[1]-view.rect.y)/view.rect.height*view.field.height);
   if(sample&&sample.kind==='entity'&&sample.radiance[3]>=.99&&sample.depth>label.anchor[2]+1e-6){hidden=true;break;}
  }
  if(hidden)continue;
  // Native MAIN uses white paper. Preserve source chroma and opacity while
  // mapping bright lettering to readable ink on that paper.
  const occurrences:NativeOccurrence[]=[],foreground=nativeLabelForeground(label.color,label.opacity);
  for(const [r,line]of lines.entries()){
   let x=Math.round((width-widths[r]!)*(label.justify===1?.5:label.justify===2?1:0));
   for(const [i,text]of [...line.text].entries()){
    const glyph=font.get(text)!;
    if(glyph.points.length)occurrences.push({id:`${label.id}/scalar-${line.offset+i}`,text,x,row:r,foreground,
     contributions:[{source:label.id,kind:'text',pixels:glyph.points.map((_,pixel)=>pixel)}]});
    x+=glyph.advance;
   }
  }
  if(!occurrences.length)continue;
  patterns.push({id:label.id,origin:[origin,row],occurrences,contacts:[],paper:[],requiredSources:[label.id]});
  supports.set(label.id,{configuration:JSON.stringify([label.text,label.justify,label.maxEmWidth,label.pivot,label.opacity,label.color,label.depthBias]),
   points:new Map([[label.id,[label.anchor[0],label.anchor[1]]]]),owners:[{reference:label.reference,source:SEMANTIC_TEXT_DOMAIN,kind:'entity'}]});
 }
 return {patterns,supports,unresolved};
}
