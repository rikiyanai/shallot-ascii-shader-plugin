import {cellKey,tileStroke,type Cell,type Point,type TrackKind,type TrainKind} from './railway';

export type Item = 'terrain' | 'lower-terrain' | TrackKind | TrainKind;
type DropAction = Item | 'move';
export interface Pick { cell:Cell; point:Point; kind:'train'|'track'|'terrain'|'empty'; id?:number; trackKind?:TrackKind }
export interface TrainSetControls {
    pick(x:number,y:number):Pick|undefined;
    plane(x:number,y:number,height:number):Point|undefined;
    paint(item:Item,cell:Cell):void;
    erase(cell:Cell,trainId?:number):void;
    move(hit:Pick,point:Point,velocity:Point):void;
    shoot(x:number,y:number):void;
    operate(cell:Cell):void;
    orbit(dx:number,dy:number,zoom:number):void;
    pan(dx:number,dy:number):void;
    reset():void;
    pause():boolean;
}
type DropGesture={
    action:DropAction;pointerId:number;owner:HTMLButtonElement;startX:number;startY:number;
    targets:Map<string,Pick>;previous?:Cell;hit?:Pick;point?:Point;velocity:Point;
    time:number;valid:boolean;dragged:boolean;cleanup:()=>void;
};

/** The canvas belongs to the camera. Edits start at an explicit palette item
 * or Move handle, and commit only on a valid release; cancellation is inert. */
export function mountTrainSetControls(canvas:HTMLCanvasElement,scene:TrainSetControls):()=>void {
    const abort=new AbortController(),signal=abort.signal;
    const pointers=new Map<number,{x:number;y:number}>();
    let cameraGesture:undefined|{x:number;y:number;dragged:boolean;select:boolean;rotate:boolean;threshold:number};
    let pairDistance=0,drop:DropGesture|undefined,selected:Pick|undefined,gunArmed=false,contextX=0,contextY=0;
    const panel=document.createElement('section');panel.id='train-set-palette';panel.setAttribute('aria-label','Train set pieces');
    panel.innerHTML='<style>'
      +'#train-set-palette{position:fixed;bottom:0;left:0;right:0;z-index:900;display:grid;gap:5px;padding:8px 10px max(8px,env(safe-area-inset-bottom));background:#fffaf0ed;color:#38291c;font:13px system-ui;box-shadow:0 -1px 8px #0002;backdrop-filter:blur(8px)}'
      +'#train-set-palette .row{display:flex;gap:5px;justify-content:center;align-items:center;flex-wrap:wrap}#train-set-palette [hidden]{display:none}'
      +'#train-set-palette button,#train-set-context button{font:inherit;min-height:42px;min-width:44px;padding:7px 10px;border:1px solid #b9a68f;border-radius:7px;background:#fffcf6;color:#38291c;touch-action:none;cursor:grab}'
      +'#train-set-palette button[data-command],#train-set-context button[data-command]{cursor:pointer}#train-set-palette button[data-dragging],#train-set-context button[data-dragging],#train-set-palette button[aria-pressed=true]{background:#d66b20;color:white;border-color:#99420b}'
      +'#train-set-palette button[data-dragging],#train-set-context button[data-dragging]{cursor:grabbing}'
      +'#train-set-context{position:fixed;z-index:1000;display:flex;gap:5px;align-items:center;flex-wrap:wrap;padding:10px;background:#fffaf0;color:#38291c;font:13px system-ui;border:1px solid #b9a68f;border-radius:9px;box-shadow:0 3px 14px #0003;width:max-content;max-width:calc(100vw - 16px)}#train-set-context[hidden]{display:none}'
      +'#train-set-help{display:block;text-align:center;font-size:12px;line-height:1.3}'
      +'#train-set-drop-preview{position:fixed;z-index:1100;pointer-events:none;padding:7px 10px;border:1px solid #99420b;border-radius:7px;background:#fff7e6;color:#38291c;font:13px system-ui;max-width:230px}'
      +'@media(max-width:600px){#train-set-palette{padding-inline:5px;gap:3px}#train-set-palette button{padding:5px 7px;font-size:12px;min-height:40px}}'
      +'</style><div class="row" data-items></div><div class="row" data-actions></div><div class="row" data-selection role="toolbar" aria-label="Selected piece actions" hidden><span data-selected></span></div><span id="train-set-help" role="status"></span><output id="train-set-status" style="text-align:center;font-size:11px"></output>'
      +'<small style="text-align:center;font-size:10px"><a href="https://sketchfab.com/3d-models/br52-steam-locomotive-animation-852f498c26ee457d933f7fbeeb50c8ab" target="_blank" rel="noopener">BR52 locomotive by Ryan.Qin</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a> · track removed; scaled for this set</small>';
    const help=panel.querySelector<HTMLSpanElement>('#train-set-help')!;
    const preview=document.createElement('output');preview.id='train-set-drop-preview';preview.hidden=true;
    const selection=panel.querySelector<HTMLElement>('[data-selection]')!;
    selection.id='train-set-context';selection.setAttribute('aria-label','Piece context menu');
    const defaultHelp='Drag to move the board · right / Alt-drag or two fingers to rotate · wheel / pinch to zoom · drag pieces from palette to build · tap for actions';
    const syncHelp=()=>{help.textContent=gunArmed?'Gun on: click / tap to shoot · drag to move the board · right / Alt-drag to rotate · wheel / pinch to zoom':defaultHelp;};
    const labels:Record<DropAction,string>={terrain:'Raise ground','lower-terrain':'Lower ground',track:'Track',switch:'Junction',station:'Station',turntable:'Turntable',steam:'Steam',passenger:'Passenger',metro:'Metro',move:'Move'};
    const positionContext=()=>{
        if(!selected)return;
        const r=selection.getBoundingClientRect(),top=panel.getBoundingClientRect().top;
        selection.style.left=(innerWidth<=600?8:Math.max(8,Math.min(innerWidth-r.width-8,contextX+12)))+'px';
        selection.style.top=Math.max(8,Math.min(top-r.height-8,innerWidth<=600?top-r.height-8:contextY+12))+'px';
    };
    const syncSelection=()=>{
        selection.hidden=!selected;
        selection.querySelector<HTMLElement>('[data-selected]')!.textContent=selected?selected.kind[0]!.toUpperCase()+selected.kind.slice(1)+' selected':'';
        route.hidden=selected?.trackKind!=='switch'&&selected?.trackKind!=='turntable';
        positionContext();
    };
    const cancelDrop=()=>{
        const old=drop;drop=undefined;
        if(old){old.cleanup();old.owner.removeAttribute('data-dragging');if(old.owner.hasPointerCapture(old.pointerId))old.owner.releasePointerCapture(old.pointerId);}
        preview.hidden=true;syncHelp();
    };
    const onBoard=(x:number,y:number)=>{
        const r=canvas.getBoundingClientRect();
        return x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom&&document.elementFromPoint(x,y)===canvas;
    };
    const stroke=(action:DropAction)=>!['steam','passenger','metro','move'].includes(action);
    const updateDrop=(event:PointerEvent)=>{
        const g=drop;if(!g||event.pointerId!==g.pointerId)return;
        g.dragged ||= Math.hypot(event.clientX-g.startX,event.clientY-g.startY)>8;
        g.valid=false;
        if(g.dragged&&onBoard(event.clientX,event.clientY)){
            if(g.action==='move'&&g.hit){
                const point=scene.plane(event.clientX,event.clientY,g.hit.point[1]+.4);
                if(point){
                    const now=performance.now(),dt=Math.max(.016,(now-g.time)/1000);
                    if(!g.point||Math.hypot(...point.map((v,i)=>v-g.point![i]!))>1e-6){
                        if(g.point)g.velocity=point.map((v,i)=>(v-g.point![i]!)/dt) as Point;
                        g.point=point;g.time=now;
                    }else if(now-g.time>150)g.velocity=[0,0,0];
                    g.valid=true;
                }
            }else{
                const pick=scene.pick(event.clientX,event.clientY);
                if(pick){
                    const cells=stroke(g.action)&&g.previous?[g.previous,...tileStroke(g.previous,pick.cell)]:[pick.cell];
                    if(!stroke(g.action))g.targets.clear();
                    for(const cell of cells){
                        const target=cellKey(...cell)===cellKey(...pick.cell)?pick:{cell,point:pick.point,kind:'empty' as const};
                        g.targets.set(cellKey(...cell),target);
                    }
                    g.previous=pick.cell;g.valid=true;
                }
            }
        }else g.previous=undefined;
        preview.hidden=false;
        preview.style.left=Math.max(4,Math.min(innerWidth-235,event.clientX+16))+'px';
        preview.style.top=Math.max(4,event.clientY-52)+'px';
        const count=stroke(g.action)?' · '+g.targets.size+' tile'+(g.targets.size===1?'':'s'):'';
        preview.textContent=labels[g.action]+count+' · '+(g.valid?'release to apply':'release to cancel');
    };
    const bindDrop=(button:HTMLButtonElement,action:DropAction)=>{
        button.addEventListener('pointerdown',event=>{
            if(event.button!==0)return;
            event.preventDefault();event.stopPropagation();cancelDrop();
            if(action==='move'&&!selected)return;
            gunArmed=false;syncGun();
            pointers.clear();cameraGesture=undefined;
            const move=(e:PointerEvent)=>{if(e.pointerId===event.pointerId){e.preventDefault();updateDrop(e);}};
            const cancel=(e:PointerEvent)=>{if(e.pointerId===event.pointerId)cancelDrop();};
            const up=(e:PointerEvent)=>{
                if(e.pointerId!==event.pointerId)return;e.preventDefault();
                updateDrop(e);const g=drop;cancelDrop();
                if(!g?.dragged||!g.valid)return;
                if(g.action==='move'&&g.hit&&g.point)scene.move(g.hit,g.point,g.velocity);
                else for(const pick of g.targets.values())scene.paint(g.action as Item,pick.cell);
                selected=undefined;syncSelection();
            };
            const cleanup=()=>{button.removeEventListener('pointermove',move);button.removeEventListener('pointerup',up);button.removeEventListener('pointercancel',cancel);button.removeEventListener('lostpointercapture',cancel);};
            drop={action,pointerId:event.pointerId,owner:button,startX:event.clientX,startY:event.clientY,targets:new Map(),hit:action==='move'?selected:undefined,velocity:[0,0,0],time:performance.now(),valid:false,dragged:false,cleanup};
            button.setAttribute('data-dragging','');button.setPointerCapture(event.pointerId);
            button.addEventListener('pointermove',move,{signal});button.addEventListener('pointerup',up,{signal});button.addEventListener('pointercancel',cancel,{signal});button.addEventListener('lostpointercapture',cancel,{signal});
            help.textContent='Drag '+labels[action].toLowerCase()+' onto the board; release to apply. Escape or release over the palette to cancel.';
        },{signal});
    };
    const item=(action:DropAction,row:string)=>{
        const b=document.createElement('button');b.type='button';b.textContent=labels[action];b.dataset.item=action;b.title='Drag '+labels[action].toLowerCase()+' onto the board';
        bindDrop(b,action);panel.querySelector(row)!.append(b);return b;
    };
    for(const action of ['terrain','lower-terrain','track','switch','station','turntable','steam','passenger','metro'] as const)item(action,'[data-items]');
    item('move','[data-selection]');
    const remove=document.createElement('button');remove.type='button';remove.dataset.command='delete';remove.textContent='Delete';
    remove.addEventListener('click',()=>{cancelDrop();if(selected){scene.erase(selected.cell,selected.kind==='train'?selected.id:undefined);selected=undefined;syncSelection();}},{signal});selection.append(remove);
    const route=document.createElement('button');route.type='button';route.dataset.command='route';route.textContent='Change route';
    route.addEventListener('click',()=>{cancelDrop();if(selected?.kind==='track')scene.operate(selected.cell);},{signal});selection.append(route);
    const dismiss=document.createElement('button');dismiss.type='button';dismiss.dataset.command='dismiss';dismiss.textContent='Deselect';
    dismiss.addEventListener('click',()=>{cancelDrop();selected=undefined;syncSelection();},{signal});selection.append(dismiss);
    const gun=document.createElement('button');gun.type='button';gun.dataset.command='gun';gun.textContent='Gun';
    const syncGun=()=>{gun.setAttribute('aria-pressed',String(gunArmed));syncHelp();};
    gun.addEventListener('click',()=>{cancelDrop();gunArmed=!gunArmed;selected=undefined;syncSelection();syncGun();},{signal});
    const pause=document.createElement('button');pause.type='button';pause.dataset.command='pause';pause.textContent='Pause trains';
    pause.addEventListener('click',()=>{cancelDrop();pause.textContent=scene.pause()?'Pause trains':'Run trains';},{signal});
    const reset=document.createElement('button');reset.type='button';reset.dataset.command='reset';reset.textContent='Reset';
    reset.addEventListener('click',()=>{cancelDrop();gunArmed=false;syncGun();selected=undefined;syncSelection();scene.reset();pause.textContent='Pause trains';},{signal});
    panel.querySelector('[data-actions]')!.append(gun,pause,reset);document.body.append(panel,selection,preview);syncGun();syncSelection();

    const start=(e:PointerEvent)=>{
        if(e.button!==0&&e.button!==1&&e.button!==2)return;e.preventDefault();e.stopImmediatePropagation();cancelDrop();canvas.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
        if(pointers.size===2){const [a,b]=[...pointers.values()];pairDistance=Math.hypot(a!.x-b!.x,a!.y-b!.y);cameraGesture=undefined;return;}
        cameraGesture={x:e.clientX,y:e.clientY,dragged:false,select:e.button===0&&!e.altKey,rotate:e.button!==0||e.altKey,threshold:e.pointerType==='touch'?8:3};
    };
    const move=(e:PointerEvent)=>{
        const previous=pointers.get(e.pointerId);if(!previous)return;e.preventDefault();e.stopImmediatePropagation();
        const dx=e.clientX-previous.x,dy=e.clientY-previous.y;pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
        if(pointers.size>=2){
            const [a,b]=[...pointers.values()],distance=Math.hypot(a!.x-b!.x,a!.y-b!.y);
            scene.orbit(dx/2,dy/2,pairDistance&&distance?pairDistance/distance:1);pairDistance=distance;return;
        }
        if(!cameraGesture)return;
        cameraGesture.dragged ||= Math.hypot(e.clientX-cameraGesture.x,e.clientY-cameraGesture.y)>cameraGesture.threshold;
        if(cameraGesture.dragged){if(cameraGesture.rotate)scene.orbit(dx,dy,1);else scene.pan(dx,dy);}
    };
    const end=(e:PointerEvent)=>{
        if(!pointers.has(e.pointerId))return;e.preventDefault();e.stopImmediatePropagation();pointers.delete(e.pointerId);
        const g=cameraGesture;cameraGesture=undefined;
        if(pointers.size||!g||e.type!=='pointerup'||g.dragged||!g.select)return;
        if(gunArmed){scene.shoot(e.clientX,e.clientY);return;}
        const pick=scene.pick(e.clientX,e.clientY);selected=pick?.kind==='empty'?undefined:pick;contextX=e.clientX;contextY=e.clientY;syncSelection();
    };
    canvas.style.touchAction='none';canvas.style.cursor='grab';
    canvas.addEventListener('pointerdown',start,{capture:true,signal});canvas.addEventListener('pointermove',move,{capture:true,signal});
    canvas.addEventListener('pointerup',end,{capture:true,signal});canvas.addEventListener('pointercancel',end,{capture:true,signal});canvas.addEventListener('lostpointercapture',end,{capture:true,signal});
    canvas.addEventListener('wheel',e=>{e.preventDefault();e.stopImmediatePropagation();cancelDrop();scene.orbit(0,0,Math.exp(Math.max(-200,Math.min(200,e.deltaY))*.002));},{capture:true,passive:false,signal});
    canvas.addEventListener('contextmenu',e=>e.preventDefault(),{signal});
    window.addEventListener('keydown',e=>{if(e.key==='Escape'){cancelDrop();gunArmed=false;syncGun();selected=undefined;syncSelection();}},{signal});
    window.addEventListener('blur',()=>{cancelDrop();pointers.clear();cameraGesture=undefined;},{signal});
    window.addEventListener('resize',positionContext,{signal});
    return ()=>{cancelDrop();abort.abort();panel.remove();selection.remove();preview.remove();canvas.style.cursor='';};
}
