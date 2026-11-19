import {DEFAULT_PLUGINS,setDefaultPlugins} from '@dylanebert/shallot';
import {mainRenderer} from '../../packages/ascii-shader/plugin.ts';
import {followWorkingTree} from '../../apps/front-door/live-reload.js';
import {OrbitMobileControlsPlugin} from './mobile-orbit-controls.ts';
// @ts-expect-error supplied by the composition's build adapter
import {gameplay} from 'virtual:shallot-example-gameplay';

// Shared build integration only. Original example compositions and controls
// remain byte-identical; applications can install mainRenderer() directly.
const renderer=mainRenderer({ascii:new URL(location.href).searchParams.get('ascii')!=='0'});
setDefaultPlugins([...DEFAULT_PLUGINS,renderer,OrbitMobileControlsPlugin,...gameplay]);

followWorkingTree();
