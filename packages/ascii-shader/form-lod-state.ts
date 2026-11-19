/**
 * The state contract for the form-table LOD writer.
 *
 * The GPU state is shared by the retained field rings of one camera/world.
 * A caller supplies the
 * actual camera/world scope and increments `epoch` when an owner lifetime can
 * be reused.  The shader stores the complete `(camera, world, epoch, owner)`
 * tuple in every state slot, so a hash collision cannot make one form inherit
 * another form's level.
 */
export type FormLodLevel=0|1|2;

export interface FormLodScope{
 /** Stable numeric identity of the camera lifetime (not a frame number). */
 readonly cameraId:number;
 /** Stable numeric identity of the world/field owner lifetime. */
 readonly worldId:number;
 /** Bumped when a source/owner index can be reused for a new lifetime. */
 readonly epoch:number;
 /** Native field pixels occupied by one exported text row. */
 readonly rowPixels:number;
 /** Explicit opt-in. False/omitted leaves the existing full producer path. */
 readonly allowCompact?:boolean;
}

export const FORM_LOD_HOLD=0.8;
export const FORM_LOD_NEAR_ENTER_ROWS=7*FORM_LOD_HOLD;
export const FORM_LOD_NEAR_LEAVE_ROWS=7/FORM_LOD_HOLD;
export const FORM_LOD_FAR_ENTER_ROWS=3*FORM_LOD_HOLD;
export const FORM_LOD_FAR_LEAVE_ROWS=3/FORM_LOD_HOLD;

/** The non-held table tier. NaN and negative extents stay conservative. */
export function formLodPlainLevel(rows:number):FormLodLevel{
 if(!Number.isFinite(rows)||rows<0)return 0;
 return rows>=7?0:rows<3?2:1;
}

/**
 * Apply separated entry/leave bands to one previously published level.
 *
 * Level 0 is near, level 1 is principal/middle and level 2 is the connected
 * far outline.  The direct 0->2/2->0 cases are intentional for a large pose
 * jump; every transition still uses the relevant entry/leave band.
 */
export function resolveFormLodLevel(rows:number,held:FormLodLevel|undefined,eligible=true):FormLodLevel{
 if(!eligible||!Number.isFinite(rows)||rows<0)return 0;
 if(held===undefined)return formLodPlainLevel(rows);
 switch(held){
  case 0:
   if(rows>=FORM_LOD_NEAR_ENTER_ROWS)return 0;
   return rows<FORM_LOD_FAR_ENTER_ROWS?2:1;
  case 1:
   if(rows>=FORM_LOD_NEAR_LEAVE_ROWS)return 0;
   return rows<FORM_LOD_FAR_ENTER_ROWS?2:1;
  case 2:
   if(rows>=FORM_LOD_NEAR_LEAVE_ROWS)return 0;
   return rows>=FORM_LOD_FAR_LEAVE_ROWS?1:2;
 }
}

/** Exact tuple comparison used by the host lifetime guard. */
export function sameFormLodScope(a:FormLodScope|undefined,b:FormLodScope|undefined):boolean{
 return !!a&&!!b&&a.cameraId===b.cameraId&&a.worldId===b.worldId&&a.epoch===b.epoch;
}

/** A disabled/default scope is deliberately unable to compact any form. */
export function conservativeFormLodScope():FormLodScope{
 return {cameraId:0,worldId:0,epoch:0,rowPixels:0,allowCompact:false};
}

/** The fixed storage ABI used by the writer-only persistent state table. */
export const FORM_LOD_STATE_WORDS=8;
export const FORM_LOD_STATE_VALID_WORD=5;
export const FORM_LOD_STATE_LEVEL_WORD=6;

export const FORM_LOD_UINT_MAX=0xffffffff;

/** Validate the ABI values before the host writes them into u32 uniforms. */
export function validateFormLodScope(scope:FormLodScope):FormLodScope{
 const ids=[['cameraId',scope.cameraId],['worldId',scope.worldId],['epoch',scope.epoch]] as const;
 for(const [name,value]of ids){
  if(!Number.isSafeInteger(value)||value<0||value>FORM_LOD_UINT_MAX){throw new RangeError(`Form LOD ${name} must be an unsigned 32-bit integer`);}
 }
 if(!Number.isFinite(scope.rowPixels)||scope.rowPixels<0){throw new RangeError('Form LOD rowPixels must be finite and non-negative');}
 if(scope.allowCompact!==undefined&&typeof scope.allowCompact!=='boolean'){throw new TypeError('Form LOD allowCompact must be boolean');}
 if(scope.allowCompact===true&&scope.rowPixels===0){throw new RangeError('Form LOD compact scope requires positive rowPixels');}
 return {...scope,cameraId:scope.cameraId>>>0,worldId:scope.worldId>>>0,epoch:scope.epoch>>>0,allowCompact:scope.allowCompact===true};
}

/** Histories are shared by all retained field rings for this camera/world. */
export function formLodScopeKey(scope:FormLodScope):string{
 const checked=validateFormLodScope(scope);return `${checked.cameraId}:${checked.worldId}`;
}

/** State word 0 means claimed but not seeded; 1..3 encode levels 0..2. */
export function encodeHeldLevel(level:FormLodLevel):number{return level+1;}
export function decodeHeldLevel(word:number):FormLodLevel|undefined{
 return word===1?0:word===2?1:word===3?2:undefined;
}

/** CPU mirror of the writer's level-word transition, used by finite tests. */
export function resolveHeldLevelWord(rows:number,word:number,eligible=true):number{
 return encodeHeldLevel(resolveFormLodLevel(rows,decodeHeldLevel(word),eligible));
}

