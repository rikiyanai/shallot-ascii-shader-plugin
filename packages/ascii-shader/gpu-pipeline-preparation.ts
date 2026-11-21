/** One owner serializes MAIN shader-module creation AND driver pipeline
 * preparation per device. A plan contains no eager GPUShaderModule: merely
 * awaiting pipeline promises after creating every module is not serial input. */
export type ComputePipelinePlan=()=>GPUComputePipelineDescriptor;
export type ComputePipelinePlans<K extends string>=ReadonlyMap<K,ComputePipelinePlan>;
/** Progress stays a scalar for existing consumers; the optional second value
 * names the keyed pipeline that is about to await the driver. */
export type ComputePipelineProgress=(fraction:number,stage?:string)=>void;
const preparationTurns=new WeakMap<GPUDevice,Promise<void>>();
const COMPILATION_INFO_TIMEOUT_MS=64;
const COMPILATION_INFO_MAX_MESSAGES=8;
const COMPILATION_INFO_SCAN_MESSAGES=256;
const COMPILATION_INFO_MAX_MESSAGE_LENGTH=512;
const COMPILATION_INFO_MAX_TOTAL_LENGTH=4096;
function diagnosticText(value:unknown,max=COMPILATION_INFO_MAX_MESSAGE_LENGTH){
 try{return String(value).slice(0,max);}catch{return '[unprintable diagnostic value]';}
}
function diagnosticNumber(value:unknown){return typeof value==='number'&&Number.isFinite(value)?String(value):'';}
function pipelineFailureMessage(cause:unknown){
 try{
  if(cause&&(typeof cause==='object'||typeof cause==='function')){
   const message=(cause as {message?:unknown}).message;
   if(typeof message==='string'&&message)return diagnosticText(message,1024);
   const name=(cause as {name?:unknown}).name;
   if(typeof name==='string'&&name)return diagnosticText(name)+' (no message)';
  }
 }catch{}
 return diagnosticText(cause,1024);
}
function shaderModuleLabel(module:unknown){
 try{
  if(module&&(typeof module==='object'||typeof module==='function')){
   const label=(module as {label?:unknown}).label;
   if(typeof label==='string'&&label)return diagnosticText(label,256);
  }
 }catch{}
 return '';
}
function compilationMessageText(message:unknown){
 try{
  if(!message||(typeof message!=='object'&&typeof message!=='function'))return '';
  const value=message as {type?:unknown;message?:unknown;lineNum?:unknown;linePos?:unknown;offset?:unknown;length?:unknown};
  const type=value.type==='error'||value.type==='warning'?value.type:'';
  if(!type)return '';
  const location=[
   diagnosticNumber(value.lineNum)&&`line ${diagnosticNumber(value.lineNum)}`,
   diagnosticNumber(value.linePos)&&`column ${diagnosticNumber(value.linePos)}`,
   diagnosticNumber(value.offset)&&`offset ${diagnosticNumber(value.offset)}`,
   diagnosticNumber(value.length)&&`length ${diagnosticNumber(value.length)}`,
  ].filter(Boolean).join(', ');
  const messageText=diagnosticText(value.message);
  return `${type}${location?` (${location})`:''}: ${messageText||'[empty message]'}`;
 }catch{return ''}
}
/** Read native compiler details only after the corresponding pipeline failed.
 * The timeout and caps keep a broken driver from replacing the original error
 * or holding startup forever. */
async function compilationDiagnostics(module:unknown){
 let getCompilationInfo:unknown;
 try{getCompilationInfo=module&&((typeof module==='object'||typeof module==='function')?(module as {getCompilationInfo?:unknown}).getCompilationInfo:undefined);}catch{return '';}
 if(typeof getCompilationInfo!=='function')return '';
 let timer:ReturnType<typeof setTimeout>|undefined;
 try{
  const info=await Promise.race([
   Promise.resolve().then(()=>getCompilationInfo.call(module)),
   new Promise<undefined>(resolve=>{timer=setTimeout(()=>resolve(undefined),COMPILATION_INFO_TIMEOUT_MS);}),
  ]);
  if(!info||(typeof info!=='object'&&typeof info!=='function'))return '';
  const messages=(info as {messages?:unknown}).messages;
  if(!Array.isArray(messages))return '';
  // Warnings emitted before the syntax error must not consume the entire
  // report. Both the scan and retained output are bounded, only on failure.
  const errors:string[]=[],warnings:string[]=[];
  for(const message of messages.slice(0,COMPILATION_INFO_SCAN_MESSAGES)){
   const line=compilationMessageText(message);if(!line)continue;
   const bucket=line.startsWith('error')?errors:warnings;
   if(bucket.length<COMPILATION_INFO_MAX_MESSAGES)bucket.push(line);
  }
  const lines:string[]=[];let total=0;
  for(const line of [...errors,...warnings].slice(0,COMPILATION_INFO_MAX_MESSAGES)){
   const next=total+(lines.length?1:0)+line.length;
   if(next>COMPILATION_INFO_MAX_TOTAL_LENGTH)break;
   lines.push(line);total=next;
  }
  return lines.join('\n');
 }catch{return '';}finally{if(timer!==undefined)clearTimeout(timer);}
}
/** Explicit synchronous inspection adapter, never a pending-runtime fallback. */
export function compileComputePipelines<K extends string>(device:GPUDevice,plans:ComputePipelinePlans<K>):ReadonlyMap<K,GPUComputePipeline>{
 if(preparationTurns.has(device))throw Error('MAIN device is preparing compute programs; await preparation before synchronous inspection');
 return new Map([...plans].map(([key,plan])=>[key,device.createComputePipeline(plan())]));
}
export async function prepareComputePipelines<K extends string>(device:GPUDevice,plans:ComputePipelinePlans<K>,progress?:ComputePipelineProgress):Promise<ReadonlyMap<K,GPUComputePipeline>>{
 const previous=preparationTurns.get(device);let release!:()=>void;
 const turn=new Promise<void>(resolve=>release=resolve);preparationTurns.set(device,turn);
 try{
  if(previous)await previous;
  const ready=new Map<K,GPUComputePipeline>();progress?.(0);
  for(const [key,plan]of plans){
   progress?.(ready.size/plans.size,key);
   let descriptor:GPUComputePipelineDescriptor|undefined;
   try{descriptor=plan();ready.set(key,await device.createComputePipelineAsync(descriptor));}
   catch(cause){
    let reason='';try{if(cause&&typeof cause==='object'&&'reason' in cause)reason=' ('+diagnosticText((cause as {reason?:unknown}).reason)+')';}catch{}
    const module=descriptor?.compute?.module;
    const label=shaderModuleLabel(module);
    const message=pipelineFailureMessage(cause);
    const details=await compilationDiagnostics(module);
    const modulePart=label?', module '+label:'';
    const compilerPart=details?'\nCompilation diagnostics:\n'+details:'';
    throw new Error('MAIN compute stage '+key+', entry '+(descriptor?.compute.entryPoint??'(not created)')+modulePart+reason+': '+message+compilerPart,{cause});
   }
   progress?.(ready.size/plans.size);
  }
  if(!plans.size)progress?.(1);
  return ready;
 }finally{release();if(preparationTurns.get(device)===turn)preparationTurns.delete(device);}
}

export interface PreparedDeviceCompute<K extends string,R>{readonly resources:R;readonly pipelines:ReadonlyMap<K,GPUComputePipeline>}
interface ComputeProgressSubscriber {
 readonly notify:ComputePipelineProgress;
 readonly failed:Promise<never>;
 fail(error:unknown):void;
}
function computeProgressSubscriber(notify:ComputePipelineProgress):ComputeProgressSubscriber{
 let reject!:((error:unknown)=>void);
 const failed=new Promise<never>((_,rejectFailure)=>{reject=rejectFailure;});
 return {notify,failed,fail:error=>reject(error)};
}
/** Pipelines/layouts have device lifetime, unlike field and frame buffers.
 * One slot owns both async runtime preparation and synchronous inspection.
 * A pending or failed async bank is never silently replaced by a sync bank. */
export class DeviceComputePrograms<K extends string,R>{
 private readonly slots=new WeakMap<GPUDevice,{spec:{resources:R;plans:ComputePipelinePlans<K>};ready?:PreparedDeviceCompute<K,R>;pending?:Promise<PreparedDeviceCompute<K,R>>;subscribers?:Set<ComputeProgressSubscriber>;currentStage?:{fraction:number;stage?:string}}>();
 constructor(private readonly plan:(device:GPUDevice)=>{resources:R;plans:ComputePipelinePlans<K>}){}
 private slot(device:GPUDevice){let slot=this.slots.get(device);if(!slot){slot={spec:this.plan(device)};this.slots.set(device,slot);}return slot;}
 async prepare(device:GPUDevice,progress?:ComputePipelineProgress):Promise<PreparedDeviceCompute<K,R>>{
  const slot=this.slot(device);
  if(slot.ready){progress?.(0);progress?.(1);return slot.ready;}
  const subscriber=progress?computeProgressSubscriber(progress):undefined;
  const subscribers=slot.subscribers??=new Set<ComputeProgressSubscriber>();
  if(subscriber){subscribers.add(subscriber);if(slot.currentStage)try{subscriber.notify(slot.currentStage.fraction,slot.currentStage.stage);}catch(error){subscribers.delete(subscriber);subscriber.fail(error);}}
  if(!slot.pending){
   slot.pending=prepareComputePipelines(device,slot.spec.plans,(fraction,stage)=>{slot.currentStage={fraction,stage};for(const current of [...subscribers])try{current.notify(fraction,stage);}catch(error){subscribers.delete(current);current.fail(error);}}).then(pipelines=>slot.ready={resources:slot.spec.resources,pipelines});
   void slot.pending.then(()=>{slot.subscribers=undefined;slot.currentStage=undefined;},()=>{slot.subscribers=undefined;slot.currentStage=undefined;});
  }else if(subscriber&&!slot.currentStage)try{subscriber.notify(0);}catch(error){subscribers.delete(subscriber);subscriber.fail(error);}
  try{
   const ready=await (subscriber?Promise.race([slot.pending,subscriber.failed]):slot.pending);
   if(subscriber)try{subscriber.notify(1);}catch(error){subscribers.delete(subscriber);subscriber.fail(error);throw error;}
   return ready;
  }catch(error){if(subscriber)subscribers.delete(subscriber);throw error;}
 }
 /** Inspection adapter only. Normal startup awaits prepare before frames. */
 get(device:GPUDevice):PreparedDeviceCompute<K,R>{
  const slot=this.slot(device);if(slot.ready)return slot.ready;
  if(slot.pending)throw Error('MAIN compute programs are not ready; await their startup preparation');
  return slot.ready={resources:slot.spec.resources,pipelines:compileComputePipelines(device,slot.spec.plans)};
 }
}
