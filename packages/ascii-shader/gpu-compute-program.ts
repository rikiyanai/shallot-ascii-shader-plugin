/** Retain unchanged function bodies reachable from one compute entry point.
 * Retained code keeps its source line numbers; no WGSL expression is
 * rewritten. This is compiler input reduction, not a rendering decision. */
export function isolateComputeEntry(source:string,entryPoint:string):string{
 const fragments:string[]=[];let from=0;
 for(let i=0;i<source.length;i++){
  if(source[i]!=='/'||(source[i+1]!=='/'&&source[i+1]!=='*'))continue;
  fragments.push(source.slice(from,i));const first=i;
  if(source[i+1]==='/'){while(i<source.length&&source[i]!=='\n')i++;}
  else{
   let depth=1;i+=2;
   while(i<source.length&&depth){
    if(source[i]==='/'&&source[i+1]==='*'){depth++;i+=2;}
    else if(source[i]==='*'&&source[i+1]==='/'){depth--;i+=2;}
    else i++;
   }
   if(depth)throw Error('Unterminated WGSL block comment');
  }
  fragments.push(source.slice(first,i).replace(/[^\n]/g,' '));from=i;i--;
 }
 fragments.push(source.slice(from));
 const code=fragments.join(''),declarations:{start:number;end:number;text:string;name?:string;fn:boolean;local:boolean}[]=[];
 let start=0,depth=0;
 for(let i=0;i<code.length;i++){
  if(code[i]==='{')depth++;
  else if(code[i]==='}')depth--;
  if(depth<0)throw Error('Unbalanced WGSL declaration');
  if(depth!==0||!(code[i]===';'||code[i]==='}'))continue;
  const text=code.slice(start,i+1),fn=/\bfn\s+(\w+)\s*\(/.exec(text),local=/\bvar\s*<\s*(?:private|workgroup)\s*>\s*(\w+)/.exec(text);
  declarations.push({start,end:i+1,text,name:fn?.[1]??local?.[1],fn:!!fn,local:!!local});start=i+1;
 }
 if(depth!==0)throw Error('Unbalanced WGSL declaration');
 const functions=new Map(declarations.filter(d=>d.fn).map(d=>[d.name!,d]));
 if(!functions.has(entryPoint))throw Error('Missing compute entry '+entryPoint);
 const live=new Set<string>(),names=new Set<string>(),pending=[entryPoint];
 while(pending.length){const name=pending.pop()!;if(live.has(name))continue;live.add(name);
  for(const match of functions.get(name)!.text.matchAll(/\b[A-Za-z_]\w*\b/g)){
   const identifier=match[0];names.add(identifier);if(functions.has(identifier)&&!live.has(identifier))pending.push(identifier);
  }
 }
 const out:string[]=[];let after=0;
 for(const d of declarations){if(d.fn?live.has(d.name!):!d.local||names.has(d.name!))continue;
  out.push(source.slice(after,d.start),source.slice(d.start,d.end).replace(/[^\n]/g,' '));after=d.end;
 }
 out.push(source.slice(after));
 return out.join('').replace(/[ \t]+$/gm,'');
}
