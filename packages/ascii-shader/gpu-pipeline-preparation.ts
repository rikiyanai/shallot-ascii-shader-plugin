/** One owner serializes MAIN shader-module creation AND driver pipeline
 * preparation per device. A plan contains no eager GPUShaderModule: merely
 * awaiting pipeline promises after creating every module is not serial input. */
export type ComputePipelinePlan=()=>GPUComputePipelineDescriptor;
export type ComputePipelinePlans<K extends string>=ReadonlyMap<K,ComputePipelinePlan>;
const preparationTurns=new WeakMap<GPUDevice,Promise<void>>();
/** Explicit synchronous inspection adapter, never a pending-runtime fallback. */
export function compileComputePipelines<K extends string>(device:GPUDevice,plans:ComputePipelinePlans<K>):ReadonlyMap<K,GPUComputePipeline>{
 if(preparationTurns.has(device))throw Error('MAIN device is preparing compute programs; await preparation before synchronous inspection');
 return new Map([...plans].map(([key,plan])=>[key,device.createComputePipeline(plan())]));
}
export async function prepareComputePipelines<K extends string>(device:GPUDevice,plans:ComputePipelinePlans<K>,progress?:(fraction:number)=>void):Promise<ReadonlyMap<K,GPUComputePipeline>>{
 const previous=preparationTurns.get(device);let release!:()=>void;
 const turn=new Promise<void>(resolve=>release=resolve);preparationTurns.set(device,turn);
 try{
  if(previous)await previous;
  const ready=new Map<K,GPUComputePipeline>();progress?.(0);
  for(const [key,plan]of plans){
   let descriptor:GPUComputePipelineDescriptor|undefined;
   try{descriptor=plan();ready.set(key,await device.createComputePipelineAsync(descriptor));}
   catch(cause){
    const reason=cause&&typeof cause==='object'&&'reason' in cause?' ('+String(cause.reason)+')':'';
    const message=cause instanceof Error?cause.message:String(cause);
    throw new Error('MAIN compute stage '+key+', entry '+(descriptor?.compute.entryPoint??'(not created)')+reason+': '+message,{cause});
   }
   progress?.(ready.size/plans.size);
  }
  if(!plans.size)progress?.(1);
  return ready;
 }finally{release();if(preparationTurns.get(device)===turn)preparationTurns.delete(device);}
}

export interface PreparedDeviceCompute<K extends string,R>{readonly resources:R;readonly pipelines:ReadonlyMap<K,GPUComputePipeline>}
/** Pipelines/layouts have device lifetime, unlike field and frame buffers.
 * One slot owns both async runtime preparation and synchronous inspection.
 * A pending or failed async bank is never silently replaced by a sync bank. */
export class DeviceComputePrograms<K extends string,R>{
 private readonly slots=new WeakMap<GPUDevice,{spec:{resources:R;plans:ComputePipelinePlans<K>};ready?:PreparedDeviceCompute<K,R>;pending?:Promise<PreparedDeviceCompute<K,R>>}>();
 constructor(private readonly plan:(device:GPUDevice)=>{resources:R;plans:ComputePipelinePlans<K>}){}
 private slot(device:GPUDevice){let slot=this.slots.get(device);if(!slot){slot={spec:this.plan(device)};this.slots.set(device,slot);}return slot;}
 async prepare(device:GPUDevice,progress?:(fraction:number)=>void):Promise<PreparedDeviceCompute<K,R>>{
  progress?.(0);const slot=this.slot(device);
  if(!slot.ready){slot.pending??=prepareComputePipelines(device,slot.spec.plans).then(pipelines=>slot.ready={resources:slot.spec.resources,pipelines});await slot.pending;}
  progress?.(1);return slot.ready!;
 }
 /** Inspection adapter only. Normal startup awaits prepare before frames. */
 get(device:GPUDevice):PreparedDeviceCompute<K,R>{
  const slot=this.slot(device);if(slot.ready)return slot.ready;
  if(slot.pending)throw Error('MAIN compute programs are not ready; await their startup preparation');
  return slot.ready={resources:slot.spec.resources,pipelines:compileComputePipelines(device,slot.spec.plans)};
 }
}
