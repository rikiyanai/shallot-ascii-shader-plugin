/** Dedicated Run.range.z flags; ordinary lineage remains in correspondence.w. */
export const COHERENT_PARENT_RUN_FLAG=0x20;
export const COHERENT_PARENT_SECONDARY_FLAG=0x40;

const COHERENT_RUN_OWNER_WGSL=/* wgsl */`
const COHERENT_PARENT_RUN_FLAG=0x20u;
const COHERENT_PARENT_SECONDARY_FLAG=0x40u;

// Native-word and source-visibility ownership. Ordinary runs do not carry the
// marker and therefore retain the existing identity.yz-only behavior. A
// coherent parent run has one primary owner in identity.yz and the optional
// second source member in correspondence.xy; the run is still emitted once.
fn coherentRunOwner(run:Run,owner:vec2u)->bool{
 let ordinary=(owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z;
 if((run.range.z&COHERENT_PARENT_RUN_FLAG)==0u){return ordinary;}
 let primary=vec2u(run.identity.y&0xdfffffffu,run.identity.z);
 let secondary=vec2u(run.correspondence.x&0xdfffffffu,run.correspondence.y);
 let normalized=vec2u(owner.x&0xdfffffffu,owner.y);
 let hasSecondary=(run.range.z&COHERENT_PARENT_SECONDARY_FLAG)!=0u;
 return all(normalized==primary)||(hasSecondary&&all(normalized==secondary));
}
`;

function replaceOne(source:string,needle:string,replacement:string):string{
 const count=source.split(needle).length-1;
 if(count!==1)throw Error(`coherent native shader anchor count ${count}: ${needle.slice(0,80)}`);
 return source.replace(needle,replacement);
}

/** Wrap the actual MAIN articulation source without importing its owner. This
 * function is deliberately pure so gpu-articulation can call it while
 * constructing its pipelines without a runtime module cycle. */
export function withCoherentNativeShader(shader:string):string{
 let coherent=shader;
 // All native source-owner checks use the same normalized owner predicate.
 // This preserves ordinary runs and lets a coherent parent retain a secondary
 // child identity without emitting that shared passage twice.
 coherent=coherent.split('((owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z)').join('coherentRunOwner(run,owner)');
 coherent=coherent.split('(owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z').join('coherentRunOwner(run,owner)');
 coherent=coherent.split('ifcoherentRunOwner(run,owner)').join('if(coherentRunOwner(run,owner))');
 coherent=coherent.split('(owner.x&0xdfffffffu)!=run.identity.y||owner.y!=run.identity.z||d<=0.0').join('!coherentRunOwner(run,owner)||d<=0.0');
 coherent=coherent.split('(owner.x&0xdfffffffu)!=run.identity.y||owner.y!=run.identity.z').join('!coherentRunOwner(run,owner)');
 return replaceOne(
  coherent,
  'struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}\n',
  'struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}\n'+COHERENT_RUN_OWNER_WGSL,
 );
}
export const COHERENT_NATIVE_CONSUMER_FUNCTION='coherentRunOwner' as const;
