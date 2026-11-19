import type {SurfaceField} from './surface-fields.ts';

const STRIDES=[4,4,8,8,8] as const;
function half(value:number):number{
 const sign=value&0x8000?-1:1,exponent=value>>10&31,mantissa=value&1023;
 return exponent===31?mantissa?NaN:sign*Infinity:sign*(exponent?1+mantissa/1024:mantissa/1024)*2**(exponent?exponent-15:-14);
}
export interface FieldSample {
 readonly source:number;readonly owner:number;readonly depth:number;
 readonly kind:'entity'|'producer-row';readonly hasUV:boolean;
 readonly uv:readonly [number,number];readonly normal:readonly [number,number,number];
 readonly radiance:readonly [number,number,number,number];
 readonly appearance?:readonly [number,number,number];
}
export interface CompletedField {
 readonly width:number;readonly height:number;readonly source:SurfaceField;
 /** Exact occupied source/owner rows. Key = source domain | producer bit31;
  * UV bit29 is not identity. Scans identities/depth only, never image tone. */
 visibleOwners?():ReadonlyMap<number,ReadonlySet<number>>;
 sample(x:number,y:number):FieldSample|undefined;
}
export interface DrawingFieldView {
 readonly pose?:{readonly eye:readonly number[];readonly forward:readonly number[];readonly focalPixels:number;readonly perspective:boolean};
 readonly field:CompletedField;readonly camera:string;
 readonly rect:{readonly x:number;readonly y:number;readonly width:number;readonly height:number};
 readonly owners:ReadonlyMap<number,number>;
 /** Unit direction toward the scene's brightest directional light at this
  * submission (the D10 band light); absent when the scene has none. */
 readonly light?:readonly number[];
 /** The scene's own fog medium at this submission (extinction density,
  * height base, height falloff); absent when the scene has none. */
 readonly fog?:readonly number[];
}
export type ConstructionFieldReadback=Pick<FieldReadback,'collect'|'completeAppearance'|'dispose'>;
/** Compiled playback consumes the retained GPU fields directly. Its source
 * graph already carries GPU visibility intervals, and native mask visibility
 * runs in featurecurves/featureplace. There is no five-channel or beauty
 * readback, no CPU whole-field walk, and no fabricated CPU pixel samples.
 * The incumbent adapter below remains available until D2 replacement review. */
export class ResidentConstructionField implements ConstructionFieldReadback {
 constructor(private readonly field:SurfaceField){}
 completeAppearance():void{}
 async collect():Promise<CompletedField>{
  return {width:this.field.width,height:this.field.height,source:this.field,
   sample:()=>{throw Error('Compiled construction fields are GPU-resident; sample visibility on the GPU');}};
 }
 dispose():void{}
}
/** One source/view submission owns all five channels. Construction may inspect
 * receiver/UV/light relationships without fitting glyphs to a camera bitmap. */
export class FieldReadback {
 private readonly buffer:GPUBuffer;
 private readonly sections:{offset:number;row:number}[]=[];
 private memory?:ArrayBuffer;
 private appearance?:{buffer:GPUBuffer;row:number;width:number;height:number;stride:number;format:GPUTextureFormat;memory?:ArrayBuffer};
 constructor(device:GPUDevice,encoder:GPUCommandEncoder,private readonly field:SurfaceField){
  let size=0;for(const stride of STRIDES){const row=Math.ceil(field.width*stride/256)*256;this.sections.push({offset:size,row});size+=row*field.height;}
  this.buffer=device.createBuffer({label:'MAIN same-submission surface fields',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  field.textures.forEach((texture,i)=>encoder.copyTextureToBuffer({texture},{buffer:this.buffer,offset:this.sections[i]!.offset,bytesPerRow:this.sections[i]!.row},[field.width,field.height]));
 }
 /** Read the native renderer's completed appearance at the SAME submission.
  * This supplies light/material/effect response on an identified receiver;
  * neither this image nor its pixels select or position glyphs. */
 completeAppearance(device:GPUDevice,encoder:GPUCommandEncoder,texture:GPUTexture,rect:DrawingFieldView['rect']):void{
  const width=Math.floor(rect.width),height=Math.floor(rect.height),format=texture.format;
  const stride=format==='rgba16float'?8:/^(?:bgra|rgba)8unorm(?:-srgb)?$/.test(format)?4:0;
  if(!stride)throw Error(`Native appearance format ${format} is unsupported`);
  const row=Math.ceil(width*stride/256)*256,buffer=device.createBuffer({label:'MAIN completed receiver appearance',size:row*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.appearance={buffer,row,width,height,stride,format};
  encoder.copyTextureToBuffer({texture,origin:[Math.floor(rect.x),Math.floor(rect.y)]},{buffer,bytesPerRow:row},[width,height]);
 }
 async collect():Promise<CompletedField>{
  await Promise.all([this.buffer.mapAsync(GPUMapMode.READ),this.appearance?.buffer.mapAsync(GPUMapMode.READ)]);this.memory=this.buffer.getMappedRange();
  if(this.appearance)this.appearance.memory=this.appearance.buffer.getMappedRange();
  const {width,height}=this.field,words=new DataView(this.memory),at=(channel:number,x:number,y:number)=>this.sections[channel]!.offset+y*this.sections[channel]!.row+x*STRIDES[channel]!;
  const appearance=this.appearance,display=appearance?.memory?new DataView(appearance.memory):undefined;
  let occupied:Map<number,Set<number>>|undefined;
  return {width,height,source:this.field,visibleOwners:()=>{
   if(occupied)return occupied;occupied=new Map();
   for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const identity=at(3,x,y),encoded=words.getUint32(identity,true);if(!(encoded&0x5fffffff)||words.getFloat32(at(0,x,y),true)<=0||!(words.getUint16(at(4,x,y)+6,true)&0x7fff))continue;
    const key=(encoded&0xdfffffff)>>>0,owner=words.getUint32(identity+4,true),rows=occupied.get(key)??new Set<number>();rows.add(owner);occupied.set(key,rows);
   }
   return occupied;
  },sample:(x,y)=>{
   x=Math.floor(x);y=Math.floor(y);if(x<0||y<0||x>=width||y>=height)return;
   const identity=at(3,x,y),encoded=words.getUint32(identity,true),source=encoded&0x5fffffff;if(!source)return;
   const normals=at(1,x,y),nx=half(words.getUint16(normals,true)),ny=half(words.getUint16(normals+2,true)),nz=1-Math.abs(nx)-Math.abs(ny);
   const n=[nx,ny,nz];if(nz<0){n[0]=(1-Math.abs(ny))*(nx<0?-1:1);n[1]=(1-Math.abs(nx))*(ny<0?-1:1);}const length=Math.hypot(...n);
   const uv=at(2,x,y),color=at(4,x,y);
   const radiance=[0,2,4,6].map(offset=>half(words.getUint16(color+offset,true))) as [number,number,number,number];
   let shown:[number,number,number]|undefined;
   if(display&&appearance){
    shown=[0,0,0];
    const px=Math.min(appearance.width-1,Math.floor((x+.5)/width*appearance.width)),py=Math.min(appearance.height-1,Math.floor((y+.5)/height*appearance.height)),address=py*appearance.row+px*appearance.stride;
    if(appearance.stride===8)for(let channel=0;channel<3;channel++)shown[channel]=half(display.getUint16(address+channel*2,true));
    else{const order=appearance.format.startsWith('bgra')?[2,1,0]:[0,1,2];for(let channel=0;channel<3;channel++)shown[channel]=display.getUint8(address+order[channel]!)/255;}
   }
   return {source,kind:encoded&0x80000000?'producer-row':'entity',hasUV:!!(encoded&0x20000000),owner:words.getUint32(identity+4,true),depth:words.getFloat32(at(0,x,y),true),normal:n.map(value=>value/length) as [number,number,number],uv:[words.getFloat32(uv,true),words.getFloat32(uv+4,true)],radiance,appearance:shown};
  }};
 }
 dispose():void{if(this.buffer.mapState==='mapped')this.buffer.unmap();this.buffer.destroy();this.memory=undefined;if(this.appearance){if(this.appearance.buffer.mapState==='mapped')this.appearance.buffer.unmap();this.appearance.buffer.destroy();this.appearance=undefined;}}
}
