/** Actual proportional-text whitespace, including the surrounding scalar
 * state. An internal gap after a padded glyph cannot pretend its trailing
 * half-space did not exist. Full spaces reset that consecutive-half state. */
export type NativeWhitespacePolicy='constrained'|'preformatted';
export function nativeSpacePlan(gap:number,previousHalf=0,leading=false,followingHalf=false,policy:NativeWhitespacePolicy='constrained'):{full:number;half:number}|undefined{
 if(!Number.isSafeInteger(gap)||gap<0||!Number.isSafeInteger(previousHalf)||previousHalf<0||policy==='constrained'&&previousHalf>2)return;
 const maximum=Math.floor(gap/11),residue=gap%5;if(maximum<residue)return;
 const full=maximum-(maximum-residue)%5,half=(gap-full*11)/5;
 if(policy==='preformatted')return {full,half};
 const initial=leading?0:2-previousHalf,capacity=initial+2*full-(followingHalf?1:0);
 if(half>capacity||leading&&full===0&&followingHalf)return;
 return{full,half};
}

/** Deterministic legal scalar spelling; never invents a gap or moves ink. */
export function nativeSpaceCodes(gap:number,previousHalf=0,leading=false,followingHalf=false,policy:NativeWhitespacePolicy='constrained'):readonly number[]|undefined{
 const plan=nativeSpacePlan(gap,previousHalf,leading,followingHalf,policy);if(!plan)return;
 if(policy==='preformatted')return [...Array<number>(plan.full).fill(0x3000),...Array<number>(plan.half).fill(32)];
 const codes:number[]=[],lead=plan.full?Math.max(plan.half-(2*plan.full-(followingHalf?1:0)),0):plan.half;
 let half=plan.half-lead;
 for(let n=0;n<lead;n++)codes.push(32);
 for(let n=0;n<plan.full;n++){
  codes.push(0x3000);const take=Math.min(half,2-(followingHalf&&n===plan.full-1?1:0));
  for(let k=0;k<take;k++)codes.push(32);half-=take;
 }
 if(half!==0)throw Error('Native whitespace plan was not completely serialized');
 return codes;
}

/** The live GPU uses this SAME two-half midrow contract, not the stricter
 * AAHub web-board collapsing rule. Return(-1,-1) when no scalar spelling exists. */
export const NATIVE_WHITESPACE_WGSL = /* wgsl */`
fn nativeSpacePlan(gap: i32, previousHalf: i32, leading: bool, followingHalf: bool) -> vec2<i32> {
  if (gap < 0 || previousHalf < 0 || previousHalf > 2) { return vec2(-1); }
  let maximum = gap / 11; let residue = gap % 5;
  if (maximum < residue) { return vec2(-1); }
  let full = maximum - (maximum - residue) % 5; let half = (gap - full * 11) / 5;
  let initial = select(2 - previousHalf, 0, leading);
  let capacity = initial + 2 * full - select(0, 1, followingHalf);
  if (half > capacity || (leading && full == 0 && followingHalf)) { return vec2(-1); }
  return vec2(full, half);
}
fn reach(g: i32, leading: bool) -> bool { return nativeSpacePlan(g, 0, leading, false).x >= 0; }
`;
