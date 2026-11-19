import {Camera,type World} from '@dylanebert/shallot';
import {Resolution} from '@dylanebert/shallot/rendering';

export const DEFAULT_NATIVE_ROWS=40;

/** One resolution owner for all MAIN cameras, including cameras added later.
 * Clearing Rows restores each camera's original engine setting, not a new
 * pixelRatio policy. Entity references prevent recycled IDs inheriting it. */
export class CellMatrix {
  private readonly originals=new Map<number,{reference:ReturnType<World['ref']>;resolution?:{width:number;height:number}}>();
  private value:number|undefined;
  private minimum=1;
  private readonly controls=new Set<HTMLInputElement>();
  get rows():number|undefined{return this.value;}
  get minimumRows():number{return this.minimum;}
  set rows(rows:number|undefined){if(rows!==undefined&&(!Number.isInteger(rows)||rows<this.minimum||rows>this.maxRows))throw Error(`Rows must be an integer from ${this.minimum} to ${this.maxRows}`);this.value=rows;for(const input of this.controls)input.value=rows===undefined?'':String(rows);}
  setMinimumRows(minimum:number):void{
    if(!Number.isSafeInteger(minimum)||minimum<1||minimum>this.maxRows)throw Error(`Rows minimum must be an integer from 1 to ${this.maxRows}`);
    this.minimum=minimum;
    for(const input of this.controls){input.min=String(minimum);input.title='Native text rows; blank restores Auto; the current HUD requires at least '+minimum+' rows.';}
    if(this.value!==undefined&&this.value<minimum){this.value=undefined;for(const input of this.controls)input.value='';}
  }
  constructor(readonly maxRows:number,value:string|null){
    if(value===null)value=String(Math.min(DEFAULT_NATIVE_ROWS,maxRows));
    const rows=Number(value);this.rows=value!==null&&value.trim()!==''&&Number.isInteger(rows)&&rows>=1&&rows<=maxRows?rows:undefined;
  }
  apply(world:World):void{
    const live=new Set<number>();
    for(const eid of world.query([Camera])){
      live.add(eid);const reference=world.ref(eid);let original=this.originals.get(eid);
      if(original?.reference!==reference){this.originals.delete(eid);original=undefined;}
      if(this.rows!==undefined){
        if(!original){
          const resolution=world.has(eid,Resolution)?{width:world.storage(Resolution).width.get(eid),height:world.storage(Resolution).height.get(eid)}:undefined;
          this.originals.set(eid,{reference,resolution});
        }
        if(!world.has(eid,Resolution))world.add(eid,Resolution,{width:0,height:this.rows*17});
        else {const resolution=world.storage(Resolution);resolution.width.set(eid,0);resolution.height.set(eid,this.rows*17);}
      }else if(original){
        if(original.resolution)world.add(eid,Resolution,original.resolution);
        else if(world.has(eid,Resolution))world.remove(eid,Resolution);
        this.originals.delete(eid);
      }
    }
    for(const [eid,original]of this.originals)if(!live.has(eid)){
      if(world.resolve(original.reference)===eid){
        if(original.resolution)world.add(eid,Resolution,original.resolution);
        else if(world.has(eid,Resolution))world.remove(eid,Resolution);
      }
      this.originals.delete(eid);
    }
  }
  control():HTMLLabelElement{
    const label=document.createElement('label'),input=document.createElement('input');input.type='number';input.min=String(this.minimum);input.max=String(this.maxRows);input.step='1';input.placeholder='Auto';input.value=this.rows===undefined?'':String(this.rows);input.style.width='5em';input.setAttribute('aria-label','Rows');input.title='Native text rows; blank restores Auto';label.append('Rows ',input);
    this.controls.add(input);
    input.addEventListener('change',()=>{
      if(!input.reportValidity())return;this.rows=input.value===''?undefined:input.valueAsNumber;
      const url=new URL(location.href);url.searchParams.set('rows',this.rows===undefined?'auto':String(this.rows));history.replaceState(history.state,'',url);
    });
    return label;
  }
}

const state={create:()=>({matrix:undefined as CellMatrix|undefined})};
/** The plugin and the physical-game adapter share this same resolution owner. */
export function cellMatrix(world:World):CellMatrix{
  const local=world.resource(state);return local.matrix??=new CellMatrix(Math.floor(world.gpu.device!.limits.maxTextureDimension2D/17),new URL(location.href).searchParams.get('rows'));
}
