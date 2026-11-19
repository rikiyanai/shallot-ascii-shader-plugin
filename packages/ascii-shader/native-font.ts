import type {NativeGlyph} from './native-page.ts';
/** Legacy banks may spell a glyph plus a literal pad as one token (`f `).
 * The page uses real Unicode scalars: lower that pad to its actual 5px advance
 * instead of inventing a second font metric or hiding it from text export. */
export function nativeScalarFont(bank:readonly NativeGlyph[]):NativeGlyph[]{
  const result=new Map<string,NativeGlyph>();
  for(const entry of bank){
    const scalars=[...entry.text];let pad=0;while(scalars.length>1&&scalars.at(-1)===' '){scalars.pop();pad+=5;}
    if(scalars.length!==1||entry.advance-pad<=0)throw Error(`Unrecognized native font token ${JSON.stringify(entry.text)}`);
    const glyph={...entry,text:scalars[0]!,advance:entry.advance-pad},previous=result.get(glyph.text);
    if(previous&&JSON.stringify(previous)!==JSON.stringify(glyph))throw Error(`Conflicting native font scalar ${glyph.text}`);
    result.set(glyph.text,glyph);
  }
  return [...result.values()];
}
