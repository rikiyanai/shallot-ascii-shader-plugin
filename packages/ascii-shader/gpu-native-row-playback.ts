import {NATIVE_ROW_ATOM_STRIDE,NATIVE_ROW_PAPER_STRIDE,NATIVE_ROW_UNIT_STRIDE} from './native-row-program-format.ts';

/** Shared near/far native-word consumer. Addresses and growth come from the
 * resident programme, never a rediscovered interval between visible glyphs.
 * Source projection/visibility remains owned by the caller's canonical span. */
export const GPU_NATIVE_ROW_PLAYBACK=/* wgsl */`
fn nativeWholeRow(record:u32,nativeRow:i32)->u32{
 let base=wholePartWordBase();if(base==NONE||nativeRow<0){return NONE;}
 return nativeRowAddress(base,grammar[record+7u],u32(nativeRow));
}
fn nativeWholeAtoms(base:u32,rowAddr:u32)->u32{
 return base+grammar[base+13u]+grammar[rowAddr+10u]*${NATIVE_ROW_ATOM_STRIDE}u;
}
fn nativeWholePapers(base:u32,rowAddr:u32)->u32{
 return base+grammar[base+14u]+grammar[rowAddr+8u]*${NATIVE_ROW_PAPER_STRIDE}u;
}
fn nativeWholeCopies(rowAddr:u32,atomAddr:u32,growth:vec4u)->u32{
 return select(1u,growth.x+1u,grammar[atomAddr]==grammar[rowAddr+18u]);
}
fn nativeWholeAtomX(record:u32,rowAddr:u32,atomAddr:u32,extra:u32,copy:u32,sourceRow:i32,sourceRows:i32)->i32{
 let base=wholePartWordBase();let nativeRow=i32(grammar[rowAddr]);
 return nativeRowAtomX(base,rowAddr,atomAddr,extra,copy)+wholePartGrowthShift(record,nativeRow,grammar[atomAddr+2u])*i32(wholePartVerticalRepeat(sourceRow,sourceRows,i32(grammar[record+5u])));
}
fn nativeWholePaperX(record:u32,nativeRow:i32,sourceRow:i32,sourceRows:i32,originalX:u32,x:i32)->i32{
 return x+wholePartGrowthShift(record,nativeRow,originalX)*i32(wholePartVerticalRepeat(sourceRow,sourceRows,i32(grammar[record+5u])));
}
// A page has at most8192 native scalar pens. Original and inserted members
// occupy disjoint ranges of a row's address; the final row has an enduring
// name rather than the changing number of middle rows above it.
fn nativeWholeMember(sourceRow:u32,sourceRows:u32,first:u32,count:u32,atom:u32,copy:u32)->u32{
 let at=first+atom*${NATIVE_ROW_ATOM_STRIDE}u;
 let row=select(sourceRow,0x3ffffu,sourceRow+1u==sourceRows&&sourceRow>0u);
 var member=2u+atom;
 if(atom==0u){member=0u;}else if(atom+1u==count){member=1u;}
 if(copy>0u){let wordAtoms=grammar[at+6u]-grammar[at+5u];member=2u+count+(copy-1u)*wordAtoms+grammar[at+1u];}
 return row*16384u+member;
}
fn nativeWholeRowFits(record:u32,whole:vec4i,sourceRow:i32)->bool{
 let base=wholePartWordBase();let partRows=i32(grammar[record+5u]);let nativeRow=wholePartNativeRow(sourceRow,whole.w,partRows);
 let rowAddr=nativeWholeRow(record,nativeRow);if(rowAddr==NONE){return false;}
 let extra=u32(whole.z)-(grammar[record+4u]&0xffffu);let growth=nativeRowGrowth(base,rowAddr,extra);if(growth.w==NONE){return false;}
 let first=nativeWholeAtoms(base,rowAddr);let count=grammar[rowAddr+11u];if(count==0u){return grammar[rowAddr+9u]==0u;}
 var previousEnd=-2147483647;
 for(var atom=0u;atom<count;atom++){
  let at=first+atom*${NATIVE_ROW_ATOM_STRIDE}u;let copies=nativeWholeCopies(rowAddr,at,growth);
  for(var copy=0u;copy<copies;copy++){
   let x=whole.x+nativeWholeAtomX(record,rowAddr,at,extra,copy,sourceRow,whole.w);let end=x+i32(grammar[at+3u]);
   if(x<0||end>i32(p.extent.x)){return false;}
   if(copy==0u){if(x<previousEnd){return false;}previousEnd=end;}
  }
 }
 let firstPaper=nativeWholePapers(base,rowAddr);let paperCount=grammar[rowAddr+9u];
 for(var paper=0u;paper<paperCount;paper++){
  let at=firstPaper+paper*${NATIVE_ROW_PAPER_STRIDE}u;let span=nativeRowPaperSpan(base,rowAddr,at,extra);if(span.y<0){return false;}
  let original=grammar[at+1u];
  let lo=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original,span.x);
  let hi=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original+grammar[at+2u],span.x+span.y);
  if(lo<0||hi<lo||hi>i32(p.extent.x)){return false;}
 }
 for(var copy=1u;copy<=growth.x;copy++){
  let span=nativeRowRepeatPaperSpan(base,rowAddr,extra,copy);if(span.y<0||span.z<0||span.w<0){return false;}
  let original=grammar[rowAddr+20u];
  let lo=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original,span.x);
  let hi=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original+grammar[rowAddr+22u],span.x+span.y);
  if(lo<0||hi<lo||hi>i32(p.extent.x)){return false;}
 }
 return true;
}
fn nativeWholeRowsFit(record:u32,whole:vec4i)->bool{
 let partRows=i32(grammar[record+5u]);if(whole.w<partRows||whole.z<i32(grammar[record+4u]&0xffffu)){return false;}
 // Linear vertical continuation is bounded by original and final copies.
 for(var row=0;row<partRows;row++){
  let sourceRow=select(row,whole.w-1,row+1==partRows);
  if(!nativeWholeRowFits(record,whole,sourceRow)){return false;}
  if(row==partRows-2&&whole.w>partRows&&!nativeWholeRowFits(record,whole,whole.w-2)){return false;}
 }
 return true;
}
fn nativeWholeMarkPaper(record:u32,whole:vec4i,sourceRow:i32,pageRow:i32){
 let base=wholePartWordBase();let nativeRow=wholePartNativeRow(sourceRow,whole.w,i32(grammar[record+5u]));
 let rowAddr=nativeWholeRow(record,nativeRow);let extra=u32(whole.z)-(grammar[record+4u]&0xffffu);
 let growth=nativeRowGrowth(base,rowAddr,extra);let first=nativeWholePapers(base,rowAddr);let count=grammar[rowAddr+9u];
 for(var paper=0u;paper<count;paper++){
  let at=first+paper*${NATIVE_ROW_PAPER_STRIDE}u;let span=nativeRowPaperSpan(base,rowAddr,at,extra);let original=grammar[at+1u];
  let lo=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original,span.x);
  let hi=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original+grammar[at+2u],span.x+span.y);
  for(var x=max(0,lo);x<min(i32(p.extent.x),hi);x++){atomicOr(&counts[paperBase()+u32(pageRow)*paperStride()+u32(x)/32u],1u<<(u32(x)%32u));}
 }
 for(var copy=1u;copy<=growth.x;copy++){
  let span=nativeRowRepeatPaperSpan(base,rowAddr,extra,copy);let original=grammar[rowAddr+20u];
  let lo=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original,span.x);
  let hi=whole.x+nativeWholePaperX(record,nativeRow,sourceRow,whole.w,original+grammar[rowAddr+22u],span.x+span.y);
  for(var x=max(0,lo);x<min(i32(p.extent.x),hi);x++){atomicOr(&counts[paperBase()+u32(pageRow)*paperStride()+u32(x)/32u],1u<<(u32(x)%32u));}
 }
}
fn nativeWholeEmit(record:u32,whole:vec4i,sourceRow:i32,pageRow:i32,runID:u32,q:vec4f,priority:u32,a:Point,b:Point,paper:bool)->u32{
 let base=wholePartWordBase();let nativeRow=wholePartNativeRow(sourceRow,whole.w,i32(grammar[record+5u]));let rowAddr=nativeWholeRow(record,nativeRow);
 let extra=u32(whole.z)-(grammar[record+4u]&0xffffu);let growth=nativeRowGrowth(base,rowAddr,extra);
 let first=nativeWholeAtoms(base,rowAddr);let count=grammar[rowAddr+11u];var emitted=0u;
 let unitFirst=base+grammar[base+12u]+grammar[rowAddr+5u]*${NATIVE_ROW_UNIT_STRIDE}u;let units=grammar[rowAddr+6u];
 for(var unit=0u;unit<units;unit++){
  let word=unitFirst+unit*${NATIVE_ROW_UNIT_STRIDE}u;
  let wordFirst=base+grammar[base+13u]+grammar[word+4u]*${NATIVE_ROW_ATOM_STRIDE}u;
  let wordCount=grammar[word+5u];let copies=nativeWholeCopies(rowAddr,wordFirst,growth);
  for(var copy=0u;copy<copies;copy++){
   for(var ordinal=0u;ordinal<wordCount;ordinal++){
    let at=wordFirst+ordinal*${NATIVE_ROW_ATOM_STRIDE}u;let atom=grammar[word+4u]-grammar[rowAddr+10u]+ordinal;
    let x=whole.x+nativeWholeAtomX(record,rowAddr,at,extra,copy,sourceRow,whole.w);
    let member=nativeWholeMember(u32(sourceRow),u32(whole.w),first,count,atom,copy);
    spanPartSourceWord(runID,member,q,x,pageRow,grammar[at+4u],1u,grammar[at+3u],priority,a,b,1u);emitted++;
   }
  }
 }
 if(paper){nativeWholeMarkPaper(record,whole,sourceRow,pageRow);}return emitted;
}
`;
