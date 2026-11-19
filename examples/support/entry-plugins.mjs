import {resolve} from 'node:path';

/** Gameplay additions are composition inputs, not runtime URL tests. Ordinary
 * scenes must not download/parse physics, weapons and three explosion banks
 * solely because the shared entry also serves the two shooting examples. */
export function exampleEntryPlugins(root,name){
  const id='virtual:shallot-example-gameplay',resolved='\0'+id;
  return {name:'shallot-example-gameplay',
    resolveId(source){if(source===id)return resolved;},
    load(source){
      if(source!==resolved)return;
      if(name!=='first-person'&&name!=='physics')return 'export const gameplay=[];';
      return `import {clickShooting} from ${JSON.stringify(resolve(root,'click-shooting.ts'))};export const gameplay=[clickShooting()];`;
    },
  };
}
