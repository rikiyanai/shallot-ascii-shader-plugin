import type {SourceWholeConstruction,SourceWholeOpening} from './source-lod.ts';
import type {CoherentParentMember,CoherentParentProduct} from './coherent-parent-union.ts';

export const COHERENT_CORRESPONDENCE_MAGIC=0x43504331;
export const COHERENT_CORRESPONDENCE_VERSION=2;
export const COHERENT_CORRESPONDENCE_LEGACY_VERSION=1;
export const COHERENT_CORRESPONDENCE_NONE=0xffffffff;
const ENTRY_WORDS=7,HEADER_WORDS=6;
export type CoherentPair=readonly [number,number];
export type CoherentCorrespondenceKind='part'|'opening'|'contact'|'protected-void'|'paper';
const kindCode=(kind:CoherentCorrespondenceKind):number=>kind==='part'?0:kind==='opening'?1:kind==='contact'?2:kind==='protected-void'?3:4;
const codeKind=(code:number):CoherentCorrespondenceKind|undefined=>code===0?'part':code===1?'opening':code===2?'contact':code===3?'protected-void':code===4?'paper':undefined;
const pairKey=(pair:CoherentPair):string=>`${pair[0]>>>0}:${pair[1]>>>0}`;
const uniquePairs=(pairs:readonly CoherentPair[]):CoherentPair[]=>{const seen=new Set<string>(),result:CoherentPair[]=[];for(const pair of pairs){const key=pairKey(pair);if(!seen.has(key)){seen.add(key);result.push([pair[0]>>>0,pair[1]>>>0]);}}return result;};
const relationLists=(value:SourceWholeConstruction):readonly (readonly [CoherentCorrespondenceKind,readonly SourceWholeOpening[]])[]=>[['opening',value.openings],['contact',value.contacts],['protected-void',value.protectedVoids],['paper',value.paper??[]]];

export interface CoherentCorrespondenceEntry {
 readonly sourceFeature:number;readonly retainedFeature:number;readonly kind:CoherentCorrespondenceKind;readonly contributors:readonly CoherentPair[];readonly relatedFeatures:readonly number[];
}
/** The exact immutable source-part order emitted beside the correspondence
 * packet.  This is the producer's ABI identity, not a feature-only lookup. */
export interface CoherentCorrespondencePart {
 readonly feature:number;readonly kind:CoherentCorrespondenceKind;readonly contributors:readonly CoherentPair[];
}
export interface CoherentCorrespondenceIndex {
 /** One entry index for each emitted source part, or NONE. */
 readonly partEntryIndices:readonly number[];
 /** One source-part index for each correspondence entry, or NONE. */
 readonly entryPartIndices:readonly number[];
 /** One entry index for each related-feature word, or NONE. */
 readonly relatedEntryIndices:readonly number[];
}
export interface DecodedCoherentCorrespondence {readonly version:number;readonly entries:readonly CoherentCorrespondenceEntry[];readonly index?:CoherentCorrespondenceIndex;}

const correspondenceIdentity=(feature:number,kind:CoherentCorrespondenceKind,pairs:readonly CoherentPair[]):string|undefined=>{
 const contributors=uniquePairs(pairs);if(contributors.length!==pairs.length)return;
 contributors.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 // The far producer deliberately shares one boundary for coincident
 // opening/protected-void/paper records. Keep all their correspondence
 // entries, but index that same emitted boundary; contacts remain distinct.
 const emissionKind=kind==='part'?'part':kind==='contact'?'contact':'boundary';
 return `${feature>>>0}|${emissionKind}|${contributors.map(pairKey).join(',')}`;
};

export function encodeCoherentCorrespondence(entries:readonly CoherentCorrespondenceEntry[],parts?:readonly CoherentCorrespondencePart[]):Uint32Array{
 const normalized=entries.map(entry=>({sourceFeature:entry.sourceFeature>>>0,retainedFeature:entry.retainedFeature>>>0,kind:entry.kind,contributors:uniquePairs(entry.contributors),relatedFeatures:[...new Set(entry.relatedFeatures.map(value=>value>>>0))]}));
 const contributorWords=normalized.reduce((sum,entry)=>sum+entry.contributors.length*2,0),relatedWords=normalized.reduce((sum,entry)=>sum+entry.relatedFeatures.length,0),version=parts===undefined?COHERENT_CORRESPONDENCE_LEGACY_VERSION:COHERENT_CORRESPONDENCE_VERSION;
 const partEntries:number[]=[],entryParts:number[]=[],relatedEntries:number[]=[];
 if(parts){
  const entriesByIdentity=new Map<string,number>(),partsByIdentity=new Map<string,number>(),features=new Map<number,number>();
  normalized.forEach((entry,index)=>{
   const identity=correspondenceIdentity(entry.retainedFeature,entry.kind,entry.contributors);
   if(identity!==undefined&&!entriesByIdentity.has(identity))entriesByIdentity.set(identity,index);
   for(const feature of [entry.sourceFeature,entry.retainedFeature])if(!features.has(feature))features.set(feature,index);
  });
  parts.forEach((part,index)=>{
   const identity=correspondenceIdentity(part.feature,part.kind,part.contributors);
   partEntries.push(identity===undefined?COHERENT_CORRESPONDENCE_NONE:entriesByIdentity.get(identity)??COHERENT_CORRESPONDENCE_NONE);
   if(identity!==undefined&&!partsByIdentity.has(identity))partsByIdentity.set(identity,index);
  });
  for(const entry of normalized){
   const identity=correspondenceIdentity(entry.retainedFeature,entry.kind,entry.contributors);
   entryParts.push(identity===undefined?COHERENT_CORRESPONDENCE_NONE:partsByIdentity.get(identity)??COHERENT_CORRESPONDENCE_NONE);
   for(const feature of entry.relatedFeatures)relatedEntries.push(features.get(feature)??COHERENT_CORRESPONDENCE_NONE);
  }
 }
 const indexWords=parts===undefined?0:partEntries.length+entryParts.length+relatedEntries.length,output=new Uint32Array(HEADER_WORDS+normalized.length*ENTRY_WORDS+contributorWords+relatedWords+indexWords);
 output.set([COHERENT_CORRESPONDENCE_MAGIC,version,normalized.length,contributorWords,relatedWords,parts?.length??0]);let contributorCursor=0,relatedCursor=0;const contributorStart=HEADER_WORDS+normalized.length*ENTRY_WORDS,relatedStart=contributorStart+contributorWords;
 normalized.forEach((entry,index)=>{const base=HEADER_WORDS+index*ENTRY_WORDS;output.set([entry.sourceFeature,entry.retainedFeature,kindCode(entry.kind),contributorCursor,entry.contributors.length,relatedCursor,entry.relatedFeatures.length],base);for(const pair of entry.contributors){output[contributorStart+contributorCursor++]=pair[0]>>>0;output[contributorStart+contributorCursor++]=pair[1]>>>0;}for(const feature of entry.relatedFeatures)output[relatedStart+relatedCursor++]=feature>>>0;});
 if(parts){const indexStart=relatedStart+relatedWords;output.set(partEntries,indexStart);output.set(entryParts,indexStart+partEntries.length);output.set(relatedEntries,indexStart+partEntries.length+entryParts.length);}
 return output;
}
export function decodeCoherentCorrespondence(words:Uint32Array):DecodedCoherentCorrespondence|undefined{
 if(words.length<HEADER_WORDS||words[0]!==COHERENT_CORRESPONDENCE_MAGIC||(words[1]!==COHERENT_CORRESPONDENCE_LEGACY_VERSION&&words[1]!==COHERENT_CORRESPONDENCE_VERSION))return;
 const entryCount=words[2]!,contributorWords=words[3]!,relatedWords=words[4]!;if(contributorWords%2!==0)return;
 const partCount=words[5]!,contributorStart=HEADER_WORDS+entryCount*ENTRY_WORDS,relatedStart=contributorStart+contributorWords,indexStart=relatedStart+relatedWords,indexWords=words[1]===COHERENT_CORRESPONDENCE_VERSION?partCount+entryCount+relatedWords:0;if(contributorStart>words.length||relatedStart>words.length||relatedStart+relatedWords>words.length||indexStart+indexWords>words.length)return;
 const entries:CoherentCorrespondenceEntry[]=[];for(let index=0;index<entryCount;index++){const base=HEADER_WORDS+index*ENTRY_WORDS,sourceFeature=words[base]!,retainedFeature=words[base+1]!,kind=codeKind(words[base+2]!);if(!kind)return;const contributorOffset=words[base+3]!,contributorsCount=words[base+4]!,relatedOffset=words[base+5]!,relatedCount=words[base+6]!;if(contributorOffset+contributorsCount*2>contributorWords||relatedOffset+relatedCount>relatedWords)return;const contributors:CoherentPair[]=[];for(let cursor=0;cursor<contributorsCount;cursor++){const at=contributorStart+contributorOffset+cursor*2;contributors.push([words[at]!,words[at+1]!]);}entries.push({sourceFeature,retainedFeature,kind,contributors,relatedFeatures:Array.from(words.slice(relatedStart+relatedOffset,relatedStart+relatedOffset+relatedCount))});}
 if(words[1]===COHERENT_CORRESPONDENCE_LEGACY_VERSION)return {version:words[1]!,entries};
 const partEntryIndices=Array.from(words.slice(indexStart,indexStart+partCount)),entryPartIndices=Array.from(words.slice(indexStart+partCount,indexStart+partCount+entryCount)),relatedEntryIndices=Array.from(words.slice(indexStart+partCount+entryCount,indexStart+indexWords));
 if(partEntryIndices.some(index=>index!==COHERENT_CORRESPONDENCE_NONE&&index>=entryCount)||entryPartIndices.some(index=>index!==COHERENT_CORRESPONDENCE_NONE&&index>=partCount)||relatedEntryIndices.some(index=>index!==COHERENT_CORRESPONDENCE_NONE&&index>=entryCount))return;
 return {version:words[1]!,entries,index:{partEntryIndices,entryPartIndices,relatedEntryIndices}};
}

function memberFeatureMap(members:readonly CoherentParentMember[]):Map<string,number[]>{const result=new Map<string,number[]>();for(const member of members)result.set(`${member.mesh>>>0}:${member.eid>>>0}`,[...new Set(member.construction.parts.map(part=>part.feature>>>0))]);return result;}
const relationKey=(kind:CoherentCorrespondenceKind,item:SourceWholeOpening):string=>`${kind}:${item.feature}:${item.contributors.map(pairKey).join(',')}:${item.geometry?.length??0}`;

type CorrespondenceItem=SourceWholeConstruction['parts'][number]|SourceWholeOpening;
type IndexedResidentItem={readonly contributors:readonly CoherentPair[]};

/**
 * Resident owner lookup is deliberately generation-local. A remapped
 * construction is immutable, so indexing it for this packet cannot observe a
 * later pose/domain generation through a stale cache. Feature alone is not an
 * identity because different copies may reuse the same feature token.
 */
const residentItemKey=(kind:CoherentCorrespondenceKind,item:CorrespondenceItem):string=>{
 const geometry=item.geometry?.map(point=>point.map(value=>String(value)).join(',')).join(';')??'-';
 const pose=item.worldFromSource===undefined?'-':Array.from(item.worldFromSource).map(value=>String(value)).join(',');
 return [kind,item.feature>>>0,item.entryRef??-1,item.context??-1,item.role??'-',item.ports.join(','),item.closed===undefined?'?':item.closed?'1':'0',item.contributors.length,geometry,pose].join('|');
};

const indexResidentItems=(resident:SourceWholeConstruction):Map<string,IndexedResidentItem[]>=>{
 const result=new Map<string,IndexedResidentItem[]>();
 const add=(kind:CoherentCorrespondenceKind,items:readonly CorrespondenceItem[])=>{
  for(const item of items){const key=residentItemKey(kind,item),bucket=result.get(key),value={contributors:item.contributors};if(bucket)bucket.push(value);else result.set(key,[value]);}
 };
 add('part',resident.parts);
 for(const [kind,items] of relationLists(resident))add(kind,items);
 return result;
};

function residentOwnerMap(product:CoherentParentProduct,resident:SourceWholeConstruction):Map<string,CoherentPair>{
 const result=new Map<string,CoherentPair>(),ambiguous=new Set<string>(),index=indexResidentItems(resident);
 const add=(source:CoherentPair,target:CoherentPair)=>{
  const key=pairKey(source),mapped:[number,number]=[target[0]>>>0,target[1]>>>0];
  if(ambiguous.has(key))return;
  const previous=result.get(key);
  if(previous&&pairKey(previous)!==pairKey(mapped)){result.delete(key);ambiguous.add(key);return;}
  result.set(key,mapped);
 };
 const map=(kind:CoherentCorrespondenceKind,source:CorrespondenceItem)=>{
  const candidates=index.get(residentItemKey(kind,source));
  // A repeated feature token with no unique source identity is unsafe. Keep
  // its incumbent owner rather than silently selecting the first copy.
  if(!candidates)return;
  if(candidates.length!==1){for(const pair of source.contributors){const key=pairKey(pair);result.delete(key);ambiguous.add(key);}return;}
  const target=candidates[0]!;
  if(target.contributors.length!==source.contributors.length){for(const pair of source.contributors){const key=pairKey(pair);result.delete(key);ambiguous.add(key);}return;}
  source.contributors.forEach((pair,index)=>{const mapped=target.contributors[index];if(mapped)add(pair,mapped);});
 };
 for(const source of product.construction.parts)map('part',source);
 for(const [kind,items] of relationLists(product.construction))for(const source of items)map(kind,source);
 return result;
}

export function coherentCorrespondenceEntries(product:CoherentParentProduct,members:readonly CoherentParentMember[],resident:SourceWholeConstruction=product.construction):CoherentCorrespondenceEntry[]{
 const memberFeatures=memberFeatureMap(members),ownerMap=residentOwnerMap(product,resident),remap=(pairs:readonly CoherentPair[])=>uniquePairs(pairs.map(pair=>ownerMap.get(pairKey(pair))??pair)),entries:CoherentCorrespondenceEntry[]=[];for(const passage of product.passages)for(const sourceFeature of passage.sourceFeatures)entries.push({sourceFeature,retainedFeature:passage.feature,kind:'part',contributors:remap(passage.contributors),relatedFeatures:passage.sourceFeatures});
 const seen=new Set<string>();for(const [kind,items] of relationLists(product.construction))for(const item of items){const key=relationKey(kind,item);if(seen.has(key))continue;seen.add(key);const related=[item.feature,...remap(item.contributors).flatMap(pair=>memberFeatures.get(pairKey(pair))??[])];entries.push({sourceFeature:item.feature,retainedFeature:item.feature,kind,contributors:remap(item.contributors),relatedFeatures:[...new Set(related)]});}
 return entries;
}
