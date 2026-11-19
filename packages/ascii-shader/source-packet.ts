/** Relation-source word ABI. V2 retires the duplicate issue count in word12;
 * complete CPU diagnostics remain in MeshSourceComplex.issues. */
export const SOURCE_PACKET_VERSION=2;
export const SOURCE_FACE_CLASS_OFFSET=12;
export const SOURCE_TWO_SIDED_OFFSET=15;
/** Fail at the worker/registration seam, before an old word12 can be read
 * as a material offset. GPU consumers receive only this registered format. */
export function assertSourcePacketVersion(words:Uint32Array):void{
 if(words.length<16||words[0]!==0x4d41494e||words[1]!==SOURCE_PACKET_VERSION)throw Error(`Unsupported MAIN source packet version ${words[1]}; expected ${SOURCE_PACKET_VERSION}`);
}
