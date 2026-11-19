import type {NativeGlyph} from './native-page.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {loadPackedRoleGrammar} from './packed-role-grammar.ts';
import preformatted from '../../assets/grammar/main-role-grammar-preformatted.v1.manifest.json' with {type:'json'};
import constrained from '../../assets/grammar/main-role-grammar-constrained.v1.manifest.json' with {type:'json'};

/** Runtime playback loads a lossless build product. It never imports the
 * grammar compiler or silently recompiles a missing/stale product on a phone. */
export function loadRuntimeRoleGrammar(font:readonly NativeGlyph[],policy:NativeWhitespacePolicy,maxWidth:number){
 const artifact=policy==='preformatted'?preformatted:policy==='constrained'?constrained:undefined;
 if(!artifact)throw Error(`Unsupported runtime grammar policy: ${policy}`);
 // Static URLs let Vite publish both products with content-addressed names,
 // including inside versioned examples and the standalone public game bundle.
 const url=policy==='preformatted'
  ?new URL('../../assets/grammar/main-role-grammar-preformatted.v1.bin.gz',import.meta.url)
  :new URL('../../assets/grammar/main-role-grammar-constrained.v1.bin.gz',import.meta.url);
 return loadPackedRoleGrammar(url,font,{policy,maxWidth,fontSha256:artifact.fontSha256,
  expectedSourceSha256:artifact.sourceSha256,expectedArtifactSha256:artifact.artifactSha256,
  expectedCompressedSha256:artifact.compressedArtifactSha256,compressed:true});
}
