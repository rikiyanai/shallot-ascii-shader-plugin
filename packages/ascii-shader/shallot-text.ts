import {type World,Text,Content,Anchor,GlobalTransform} from '@dylanebert/shallot';
import {composeGlobalTransform} from '@dylanebert/shallot/transform';
import {renderLayers} from '@dylanebert/shallot/rendering';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativeLabel} from './semantic-text.ts';

export function collectNativeLabels(world:World,camera:number,clip:Float32Array,rect:DrawingFieldView['rect']):NativeLabel[]{
 const result:NativeLabel[]=[],transform=new Float32Array(16),labels=world.storage(Text),layers=renderLayers(world,camera);
 for(const eid of world.query([Text,GlobalTransform])){
  if(!(renderLayers(world,eid)&layers))continue;
  const text=world.resource(Content).name(labels.content.get(eid));if(!text||labels.opacity.get(eid)<=0)continue;
  composeGlobalTransform(world,eid,transform);
  const p=[transform[12]!,transform[13]!,transform[14]!,1];
  const projected=[0,1,2,3].map(row=>p.reduce((sum,value,column)=>sum+value*clip[column*4+row]!,0));
  if(projected[3]!<=0)continue;
  const ndc=projected.map(value=>value/projected[3]!);if(ndc[2]!<0||ndc[2]!>1)continue;
  const reference=world.ref(eid),anchored=world.has(eid,Anchor),pivot=world.storage(Anchor).offset;
  result.push({id:`camera-${world.ref(camera)}/text/${reference}`,reference,text,camera:`camera-${world.ref(camera)}`,
   anchor:[rect.x+(ndc[0]!+1)*rect.width/2,rect.y+(1-ndc[1]!)*rect.height/2,ndc[2]!],
   pivot:anchored?[pivot.x.get(eid)+.5,.5-pivot.y.get(eid)]:[0,1],
   justify:labels.justify.get(eid),maxEmWidth:labels.maxWidth.get(eid)/labels.fontSize.get(eid),depthBias:labels.depthBias.get(eid),
   opacity:labels.opacity.get(eid),color:labels.color.get(eid)});
 }
 return result;
}
