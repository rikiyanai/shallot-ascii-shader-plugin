import type {ResidentIndexedIncidence} from './gpu-source-incidence.ts';

/** The only compiler input needed for a GPU-produced indexed source. The
 * producer's index buffer is already resident; this adapter does not receive
 * or allocate CPU positions/indices. */
export interface GpuIndexedSourceCompileInput {
 readonly indices:GPUBufferBinding;
 readonly vertexCapacity:number;
 readonly indexCount:number;
 readonly edgeCapacity?:number;
 readonly indexPattern?:'independent-quads';
}
export interface GpuIndexedSourceCompiler {build(indices:GPUBufferBinding,vertices:number,corners:number,edgeCapacity?:number,indexPattern?:'independent-quads'):Promise<ResidentIndexedIncidence>}

export function compileGpuIndexedSource(compiler:GpuIndexedSourceCompiler,input:GpuIndexedSourceCompileInput):Promise<ResidentIndexedIncidence>{
 if(!Number.isSafeInteger(input.vertexCapacity)||input.vertexCapacity<1)throw Error('GPU indexed source compiler requires a positive vertex capacity');
 if(!Number.isSafeInteger(input.indexCount)||input.indexCount<3||input.indexCount%3)throw Error('GPU indexed source compiler requires whole triangles');
 if(input.edgeCapacity!==undefined&&(!Number.isSafeInteger(input.edgeCapacity)||input.edgeCapacity<1||input.edgeCapacity>input.indexCount))throw Error('GPU indexed source compiler edge capacity is invalid');
 const available=input.indices.size??input.indices.buffer.size-(input.indices.offset??0);
 if(available<input.indexCount*4)throw Error('GPU indexed source compiler index binding is too short');
 return compiler.build(input.indices,input.vertexCapacity,input.indexCount,input.edgeCapacity??input.indexCount,input.indexPattern);
}
