                                                                               
                                                                               
                                                                               
                                                                        
                                                             
                                                                        
                                                                              
                                                                   
                                                                        
export const PRIMARY_IMAGE_MARKER='/* TRUEFF_MAIN_PRIMARY_IMAGE_0117 */';
function once(text,from,to,label){if(text.split(from).length!==2)throw Error('MAIN primary image: '+label+' changed');return text.replace(from,to);}
export function transformPrimaryImageTonemapping(tonemap){
 if(tonemap.includes(PRIMARY_IMAGE_MARKER))return tonemap;
 tonemap=once(tonemap,'            if (!view?.present || !view.framebuffer || world.has(eid, CustomPresentation)) continue;',
  '            if (!view?.present || !view.framebuffer || world.has(eid, CustomPresentation) || primaryOwnsView(world, view)) continue;','tonemapping camera loop');
 return `${PRIMARY_IMAGE_MARKER}\nimport { primaryOwnsView } from "./phase-geometry";\n`+tonemap;
}
export function transformPrimaryImageOutline(outline){
 if(outline.includes(PRIMARY_IMAGE_MARKER))return outline;
 outline=once(outline,'            const view = world.resource(Views).get(camEid);\n            if (!view?.framebuffer) continue;\n            const cameraMask',
  '            const view = world.resource(Views).get(camEid);\n            if (!view?.framebuffer || primaryOwnsView(world, view)) continue;\n            const cameraMask','outline camera loop');
 return `${PRIMARY_IMAGE_MARKER}\nimport { primaryOwnsView } from "../../core/rendering/phase-geometry";\n`+outline;
}
