import {WgslReflect,type TypeInfo,type StructInfo,type FunctionInfo} from 'wgsl_reflect';

interface Attribute {name:string;value:unknown}
type Attributes=readonly Attribute[]|null;
const attributes=(list:Attributes)=>list?.map(a=>`@${a.name}${a.value===null?'':`(${Array.isArray(a.value)?a.value.join(','):a.value})`}`).join(' ')??'';
const value=(list:Attributes,name:string)=>list?.find(a=>a.name===name)?.value;
const typeName=(type:TypeInfo)=>type.getTypeName();
const members=(type:TypeInfo)=>type.isStruct?(type as StructInfo).members:undefined;

function ordinaryFunction(code:string,entry:FunctionInfo):string{
 const start=new RegExp(`@${entry.stage}\\s+fn\\s+${entry.name}\\s*\\(`).exec(code);
 if(!start)throw Error(`Native field cannot find ${entry.stage} entry ${entry.name}`);
 let at=start.index+start[0].length,depth=1;
 for(;at<code.length&&depth;at++){if(code[at]==='(')depth++;else if(code[at]===')')depth--;}
 if(depth)throw Error('Native field entry has an incomplete argument list');
 const body=code.indexOf('{',at);if(body<0)throw Error('Native field entry has no body');
 const args=entry.arguments.map(a=>`${a.name}:${typeName(a.type)}`).join(',');
 return code.slice(0,start.index)+`fn ${entry.name}(${args})${entry.returnType?`->${typeName(entry.returnType)}`:''}`+code.slice(body);
}

export interface FieldStage {
 readonly vertex:string;readonly fragment:string;readonly group:number;
 readonly ownerKind:'entity'|'producer-row'|'mixed';readonly hasUV:boolean;
 readonly resources:ReadonlyMap<number,ReadonlySet<number>>;
}

/** Execute the native material's actual vertex AND fragment functions. The
 * resulting field is geometry/appearance input to construction, never a text
 * filter. Original discard, UV animation and material lighting stay executable.
 * Five targets consume exactly the core 32-byte color-attachment budget.
 */
export function fieldStage(vertexCode:string,fragmentCode:string,vertexEntry?:string,fragmentEntry?:string,emissive=false,primary=false):FieldStage{
 const vr=new WgslReflect(vertexCode),fr=new WgslReflect(fragmentCode);
 const vertex=vr.entry.vertex.filter(e=>!vertexEntry||e.name===vertexEntry),fragment=fr.entry.fragment.filter(e=>!fragmentEntry||e.name===fragmentEntry);
 if(vertex.length!==1||fragment.length!==1)throw Error('Native field requires the resolved vertex and fragment entries');
 const v=vertex[0]!,f=fragment[0]!;
 const usedGroups=new Set([vr,fr].flatMap(reflection=>[...reflection.uniforms,...reflection.storage,...reflection.textures,...reflection.samplers].map(resource=>resource.group)));
 const group=[0,1,2,3].find(id=>!usedGroups.has(id));if(group===undefined)throw Error('Native field needs one free uniform bind group');
 const locations=new Set(v.outputs.filter(o=>o.locationType==='location').map(o=>Number(o.location)));
 const ownerLocation=Array.from({length:16},(_,i)=>i).find(i=>!locations.has(i));
 if(ownerLocation===undefined)throw Error('Native field has no free inter-stage source identity slot');
 if(!v.returnType)throw Error('Native vertex has no output');
 const out=members(v.returnType);
 const fields=out??[{name:'mainPosition',type:v.returnType,attributes:v.returnType.attributes}];
 const position=fields.find(m=>value(m.attributes,'builtin')==='position');
 if(!position)throw Error('Native field vertex has no clip position');
 const eid=fields.find(m=>m.name==='eid'&&typeName(m.type)==='u32');
 const row=fields.find(m=>m.name==='row'&&typeName(m.type)==='u32');
 let instance:string|undefined;
 for(const arg of v.arguments){
  if(value(arg.attributes,'builtin')==='instance_index')instance=arg.name;
  for(const member of members(arg.type)??[])if(value(member.attributes,'builtin')==='instance_index')instance=`${arg.name}.${member.name}`;
 }
 const declarations=v.arguments.map(arg=>`${attributes(arg.attributes)} ${arg.name}:${typeName(arg.type)}`);
 if(!instance){instance='mainInstance';declarations.push('@builtin(instance_index) mainInstance:u32');}
 const vsource=ordinaryFunction(vertexCode,v)+`
struct MainFieldVertex {
 ${fields.map(field=>`${attributes(field.attributes)} ${field.name}:${typeName(field.type)},`).join('\n')}
 @location(${ownerLocation}) @interpolate(flat) mainOwner:vec2u,
}
@vertex fn mainFieldVertex(${declarations.join(',')})->MainFieldVertex{
 let native=${v.name}(${v.arguments.map(arg=>arg.name).join(',')});
 // Standard material output carries BOTH ECS and procedural rows. The eid
 // sentinel is per invocation, not a shader-wide ownership classification.
 let mainSourceProducer=${eid?`native.${eid.name}==0xffffffffu`:'true'};
 let mainSourceRow=${row?`native.${row.name}`:instance};
 let mainSourceOwner=${eid?`select(native.${eid.name},mainSourceRow,mainSourceProducer)`:'mainSourceRow'};
 return MainFieldVertex(${fields.map(field=>out?`native.${field.name}`:'native').join(',')},vec2u(mainSourceOwner,select(0u,0x80000000u,mainSourceProducer)));
}
`;
 const inputs:{name:string;type:TypeInfo;attributes:Attributes;original:string}[]=[],arguments_:string[]=[];
 for(const arg of f.arguments){
  const fields=members(arg.type);
  if(fields){const args:string[]=[];for(const member of fields){const name=`main_${arg.name}_${member.name}`;inputs.push({name,type:member.type,attributes:member.attributes,original:member.name});args.push(`input.${name}`);}arguments_.push(`${typeName(arg.type)}(${args.join(',')})`);}
  else{const name=`main_${arg.name}`;inputs.push({name,type:arg.type,attributes:arg.attributes,original:arg.name});arguments_.push(`input.${name}`);}
 }
 let fragPosition=inputs.find(input=>value(input.attributes,'builtin')==='position')?.name;
 if(!fragPosition){fragPosition='mainPosition';inputs.push({name:fragPosition,type:position.type,attributes:[{name:'builtin',value:'position'}],original:'position'});}
 const normal=inputs.find(input=>input.original==='worldNormal'&&/^vec3/.test(typeName(input.type)));
 const uv=inputs.find(input=>input.original==='uv'&&/^vec2/.test(typeName(input.type)));
 const returnFields=f.returnType?members(f.returnType):undefined;
 const color=returnFields?.find(member=>String(value(member.attributes,'location'))==='0');
 const directColor=f.returnType&&!returnFields&&String(value(f.returnType.attributes,'location'))==='0';
 const depth=returnFields?.find(member=>value(member.attributes,'builtin')==='frag_depth');
 const sampleMask=returnFields?.find(member=>value(member.attributes,'builtin')==='sample_mask');
 if(!color&&!directColor)throw Error('Native field fragment has no color output');
 const fsource=ordinaryFunction(fragmentCode,f)+`
struct MainFieldParameters {worldFromClip:mat4x4f,extent:vec4f,info:vec4u}
@group(${group}) @binding(0) var<uniform> mainField:MainFieldParameters;
 ${primary&&emissive?'@group('+group+') @binding(1) var<storage,read_write> mainCoverage:array<atomic<u32>>;':''}
struct MainFieldInput {
 ${inputs.map(input=>`${attributes(input.attributes)} ${input.name}:${typeName(input.type)},`).join('\n')}
 @location(${ownerLocation}) @interpolate(flat) mainOwner:vec2u,
}
struct MainFieldOutput {
 @location(0) depth:f32,
 @location(1) normal:vec2f,
 @location(2) uv:vec2f,
 @location(3) owner:vec2u,
 @location(4) radiance:vec4f,
 ${depth?'@builtin(frag_depth) nativeDepth:f32,':''}
 ${sampleMask?'@builtin(sample_mask) nativeSampleMask:u32,':''}
}
fn mainFieldWorld(position:vec4f)->vec3f{
 let clip=vec4f(position.xy/mainField.extent.xy*vec2f(2,-2)+vec2f(-1,1),position.z,1);
 let p=mainField.worldFromClip*clip;if(abs(p.w)<1e-20){return vec3f(0);}return p.xyz/p.w;
}
fn mainFieldOct(value:vec3f)->vec2f{
 let n=value/max(dot(abs(value),vec3f(1)),1e-20);
 return select((vec2f(1)-abs(n.yx))*select(vec2f(-1),vec2f(1),n.xy>=vec2f(0)),n.xy,n.z>=0);
}
@fragment fn mainFieldFragment(input:MainFieldInput)->MainFieldOutput{
 let world=mainFieldWorld(input.${fragPosition});
 let geometricNormal=cross(dpdx(world),dpdy(world));
 let native=${f.name}(${arguments_.join(',')});
 let color=vec4f(${color?`native.${color.name}`:'native'});
 ${primary&&emissive?`// Coverage has a separate bitset: native RGB/alpha and blend remain real.
 if(max(color.r,max(color.g,color.b))>.05){
  let mainPixel=vec2u(input.${fragPosition}.xy);
  let mainBit=mainPixel.y*u32(mainField.extent.x)+mainPixel.x;
  atomicOr(&mainCoverage[1u+(mainBit>>5u)],1u<<(mainBit&31u));
 }
 return MainFieldOutput(0.0,vec2f(0),vec2f(0),vec2u(0),color${depth?`,native.${depth.name}`:''}${sampleMask?`,native.${sampleMask.name}`:''});
 ` : emissive?`// EMISSIVE (an additive draw: tone2 particles). Coverage only: the
 // pipeline writes the radiance alpha (+4) and nothing else.
 if(max(color.r,max(color.g,color.b))<=.05){discard;}
 return MainFieldOutput(0.0,vec2f(0),vec2f(0),vec2u(0),vec4f(0,0,0,4)${depth?`,native.${depth.name}`:''}${sampleMask?`,native.${sampleMask.name}`:''});`:`if(color.a<=0){discard;}
 return MainFieldOutput(${depth?`native.${depth.name}`:`input.${fragPosition}.z`},mainFieldOct(${normal?`input.${normal.name}`:'geometricNormal'}),${uv?`input.${uv.name}`:'vec2f(0)'},vec2u(mainField.info.x|input.mainOwner.y,input.mainOwner.x),color${depth?`,native.${depth.name}`:''}${sampleMask?`,native.${sampleMask.name}`:''});`}
}
`;
 const resources=new Map<number,Set<number>>();
 for(const entry of [new WgslReflect(vsource).entry.vertex.find(e=>e.name==='mainFieldVertex')!,new WgslReflect(fsource).entry.fragment.find(e=>e.name==='mainFieldFragment')!])for(const resource of entry.resources){const bindings=resources.get(resource.group)??new Set<number>();bindings.add(resource.binding);resources.set(resource.group,bindings);}
 return {vertex:vsource,fragment:fsource,group,ownerKind:eid?'mixed':'producer-row',hasUV:!!uv,resources};
}
