import {DEFAULT_PLUGINS,setDefaultPlugins} from '@dylanebert/shallot';
import {mainRenderer} from '../../packages/ascii-shader/plugin.ts';
import {followWorkingTree} from '../../apps/front-door/live-reload.js';
import {OrbitMobileControlsPlugin} from './mobile-orbit-controls.ts';
// Every imported example uses the same Tamanegi/Refrog shell. This is UI
// typography and chrome only; the scene canvas and MAIN native glyph font are
// left to their existing owners. Import the canonical Xanh Mono stylesheet so
// Train set cannot install its own system-font palette over the shared runner.
import '../../apps/chapel/dist/style.css';
import './shared-ui.css';
// @ts-expect-error supplied by the composition's build adapter
import {gameplay} from 'virtual:shallot-example-gameplay';

const routeParts=location.pathname.split('/').filter(Boolean);
const isTrainSet=routeParts.some((part,index)=>part==='train-set'&&routeParts[index-1]==='examples');

// Train's authored controls append their help, counters and model credit to
// the same fixed palette.  Keep the imported composition and its controls
// byte-identical: the shared presentation boundary removes only the bulky
// supplemental copy from the gameplay viewport and preserves the attribution
// in a closed, keyboard/touch-accessible disclosure.
const installTrainSetCredits=()=>{
    const panel=document.getElementById('train-set-palette');
    if(!panel)return false;
    const credit=[...panel.children].find(node=>node.tagName==='SMALL');
    if(!credit)return Boolean(document.getElementById('train-set-credits'));
    let details=document.getElementById('train-set-credits');
    if(!details){
        details=document.createElement('details');details.id='train-set-credits';
        const summary=document.createElement('summary');summary.textContent='Credits';
        details.append(summary);document.body.append(details);
    }
    details.append(credit);
    return true;
};
const observeTrainSetCredits=()=>{
    if(!document.body){document.addEventListener('DOMContentLoaded',observeTrainSetCredits,{once:true});return;}
    const apply=()=>{if(installTrainSetCredits())observer.disconnect();};
    const observer=new MutationObserver(apply);
    observer.observe(document.body,{childList:true,subtree:true});
    apply();
};
if(isTrainSet&&typeof document!=='undefined')observeTrainSetCredits();


// glyph or scene branch.  MAIN's CellMatrix keeps an explicit ?rows= choice
// authoritative, including ?rows=auto; every other example keeps its normal
// default.
// Shared build integration only. Original example compositions and controls
// remain byte-identical; applications can install mainRenderer() directly.
const renderer=mainRenderer({ascii:new URL(location.href).searchParams.get('ascii')!=='0',rows:isTrainSet?130:undefined});
setDefaultPlugins([...DEFAULT_PLUGINS,renderer,OrbitMobileControlsPlugin,...gameplay]);

followWorkingTree();
