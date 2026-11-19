import type {World} from '../../engine';
import type {PhaseRenderer} from './phases';
import type {View} from './view';
import type {RenderPipelineRecipe} from '../../engine/runtime/pipeline-source';

export interface PhaseGeometryDraw {
  readonly renderer:PhaseRenderer;
  readonly stage:'opaque'|'after-opaque'|'transparent';
  readonly stableId:number;
  readonly stablePart:number;
  readonly ordinal:number;
  readonly pipeline:GPURenderPipeline|undefined;
  readonly recipe?:RenderPipelineRecipe;
  readonly groups:readonly {index:number;group:GPUBindGroup;offsets:readonly number[]}[];
  readonly vertices:readonly {slot:number;buffer:GPUBuffer;offset:number;size?:number}[];
  readonly index?:{buffer:GPUBuffer;format:GPUIndexFormat;offset:number;size?:number};
  readonly viewport?:readonly number[];
  readonly scissor?:readonly number[];
  readonly blendConstant?:GPUColor;readonly stencilReference?:number;
  readonly command:'draw'|'draw-indexed'|'draw-indirect'|'draw-indexed-indirect'|'bundles';
  readonly arguments:readonly (number|GPUBuffer|GPURenderBundle[])[];
}
/** Opt-in observation of actual native draws. Recording never reissues a draw,
 * changes its state, or guesses an emitter from an example or resource name. */
export const PhaseGeometryCapture={create:()=>({enabled:false,includeStandard:false,views:new WeakMap<View,PhaseGeometryDraw[]>(),


  isPrimary:undefined as undefined|(()=>boolean),
  primary:undefined as undefined|((world:World,view:View,eid:number)=>(()=>void)|undefined)})};

export function captureGeometryPass(world:World,view:View,native:GPURenderPassEncoder|undefined):{
  pass:GPURenderPassEncoder;
  source:(renderer:PhaseRenderer,stage:PhaseGeometryDraw['stage'],stableId?:number,stablePart?:number)=>void;
}{
  const capture=world.resource(PhaseGeometryCapture);
  if(!capture.enabled){if(!native)throw Error('Capture-only pass has no primary producer');return{pass:native,source:()=>{}};}
  const records:PhaseGeometryDraw[]=[];capture.views.set(view,records);
  let context:{renderer:PhaseRenderer;stage:PhaseGeometryDraw['stage'];stableId:number;stablePart:number;ordinal:number}|undefined;
  let pipeline:GPURenderPipeline|undefined,recipe:RenderPipelineRecipe|undefined,index:PhaseGeometryDraw['index'],viewport:readonly number[]|undefined,scissor:readonly number[]|undefined;
  const groups=new Map<number,{index:number;group:GPUBindGroup;offsets:readonly number[]}>();
  const vertices=new Map<number,{slot:number;buffer:GPUBuffer;offset:number;size?:number}>();
  let blendConstant:GPUColor|undefined,stencilReference:number|undefined;
  const selectPipeline=()=>{if(native){if(recipe)native.setPipeline(recipe.materialize());else if(pipeline)native.setPipeline(pipeline);}};
  const record=(command:PhaseGeometryDraw['command'],args:PhaseGeometryDraw['arguments'])=>{
    if(!context)throw Error('Geometry draw has no phase owner');
    if(!capture.includeStandard&&context.renderer.geometryAdapter==='standard-material')return;
    records.push({...context,pipeline,recipe,groups:[...groups.values()].map(g=>({...g,offsets:[...g.offsets]})),vertices:[...vertices.values()],index,viewport,scissor,blendConstant,stencilReference,command,arguments:args});
    context.ordinal++;
  };
  const methods=new Map<PropertyKey,unknown>();
  const pass=new Proxy(native??({} as GPURenderPassEncoder),{get(target,key){
    if(methods.has(key))return methods.get(key);
    let value:unknown;
    if(key===Symbol.for('shallot.phase.render-recipe'))value=(next:RenderPipelineRecipe)=>{recipe=next;pipeline=undefined;selectPipeline();};
    else if(key==='setPipeline')value=(next:GPURenderPipeline)=>{recipe=undefined;pipeline=next;selectPipeline();};
    else if(key==='setBindGroup')value=(slot:number,group:GPUBindGroup|null,offsets:Iterable<number>|Uint32Array=[],start?:number,length?:number)=>{
      let values=Array.from(offsets);
      if(start!==undefined){
        if(!(offsets instanceof Uint32Array)||!Number.isInteger(start)||start<0||!Number.isInteger(length)||length!<0||start+length!>values.length)throw RangeError('Invalid native dynamic-offset range');
        values=values.slice(start,start+length!);
      }
      if(group)groups.set(slot,{index:slot,group,offsets:values});else groups.delete(slot);
      native?.setBindGroup(slot,group,values);
    };
    else if(key==='setVertexBuffer')value=(slot:number,buffer:GPUBuffer|null,offset=0,size?:number)=>{if(buffer)vertices.set(slot,{slot,buffer,offset,size});else vertices.delete(slot);native?.setVertexBuffer(slot,buffer,offset,size);};
    else if(key==='setIndexBuffer')value=(buffer:GPUBuffer,format:GPUIndexFormat,offset=0,size?:number)=>{index={buffer,format,offset,size};native?.setIndexBuffer(buffer,format,offset,size);};
    else if(key==='setViewport')value=(...args:Parameters<GPURenderPassEncoder['setViewport']>)=>{viewport=[...args];native?.setViewport(...args);};
    else if(key==='setScissorRect')value=(...args:Parameters<GPURenderPassEncoder['setScissorRect']>)=>{scissor=[...args];native?.setScissorRect(...args);};
    else if(key==='draw')value=(vertices:number,instances=1,firstVertex=0,firstInstance=0)=>{record('draw',[vertices,instances,firstVertex,firstInstance]);native?.draw(vertices,instances,firstVertex,firstInstance);};
    else if(key==='drawIndexed')value=(indices:number,instances=1,firstIndex=0,baseVertex=0,firstInstance=0)=>{record('draw-indexed',[indices,instances,firstIndex,baseVertex,firstInstance]);native?.drawIndexed(indices,instances,firstIndex,baseVertex,firstInstance);};
    else if(key==='drawIndirect')value=(buffer:GPUBuffer,offset:number)=>{record('draw-indirect',[buffer,offset]);native?.drawIndirect(buffer,offset);};
    else if(key==='drawIndexedIndirect')value=(buffer:GPUBuffer,offset:number)=>{record('draw-indexed-indirect',[buffer,offset]);native?.drawIndexedIndirect(buffer,offset);};
    else if(key==='setBlendConstant')value=(color:GPUColor)=>{blendConstant=Array.isArray(color)?[...color]:{...color};native?.setBlendConstant(color);};
    else if(key==='setStencilReference')value=(reference:number)=>{stencilReference=reference;native?.setStencilReference(reference);};
    else if(key==='executeBundles')value=(bundles:Iterable<GPURenderBundle>)=>{const list=Array.from(bundles);record('bundles',[list]);native?.executeBundles(list);};
    else if(!native&&['pushDebugGroup','popDebugGroup','insertDebugMarker','end'].includes(String(key)))value=()=>{};
    else{const member=Reflect.get(target,key,target);if(typeof member!=='function')return member;value=member.bind(target);}
    methods.set(key,value);return value;
  },set(target,key,value){return Reflect.set(target,key,value,target);}});
  return{pass,source:(renderer,stage,stableId=0,stablePart=0)=>{context={renderer,stage,stableId,stablePart,ordinal:0};selectPipeline();}};
}
