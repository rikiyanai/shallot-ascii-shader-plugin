/**
 * Engine-neutral mobile input surface.
 *
 * This module owns only DOM input and normalized intent. It never imports an
 * engine, dispatches keyboard events, changes a camera/actor pose, or claims
 * the canvas's right-side look gesture.
 */

export interface MobileActionSpec {
  readonly label?: string;
  readonly ariaLabel?: string;
  readonly onPress: (event: PointerEvent) => void;
  /** Optional matching end callback for held actions such as Attack. */
  readonly onRelease?: (event: PointerEvent) => void;
  /** Called instead of onRelease when the pointer is cancelled or ownership is lost. */
  readonly onCancel?: () => void;
}

export type MobileAction = MobileActionSpec | ((event: PointerEvent) => void);

export interface MobileControlsOptions {
  readonly parent?: HTMLElement;
  readonly active: boolean | (() => boolean);
  readonly onMove: (x: number, y: number) => void;
  readonly actions?: Readonly<Record<string, MobileAction>>;
  /** Override matchMedia('(pointer: coarse)') for hosts and deterministic tests. */
  readonly coarsePointer?: boolean;
  readonly deadzone?: number;
  /** Reserve space for an existing composition's bottom toolbar. */
  readonly bottomInset?: number;
  /** Cluster flight actions beside the stick, leaving the right side for look/fire. */
  readonly actionAlign?: 'start' | 'end';
}

export interface MobileControlsHandle {
  readonly element: HTMLElement;
  readonly joystick: HTMLElement;
  readonly knob: HTMLElement;
  readonly actionButtons: Readonly<Record<string, HTMLButtonElement>>;
  readonly mounted: boolean;
  refresh(): void;
  reset(): void;
  setActive(active: boolean): void;
  dispose(): void;
}

const DEFAULT_DEADZONE = 0.16;
const CONTROL_SIZE = 76;
const KNOB_SIZE = 32;
const TOUCH_TARGET = 44;

type ListenerTarget = {
  addEventListener(type: string, listener: EventListener, options?: AddEventListenerOptions | boolean): void;
  removeEventListener(type: string, listener: EventListener, options?: EventListenerOptions | boolean): void;
};

type PointerLike = PointerEvent & { readonly pointerId: number; readonly clientX: number; readonly clientY: number };

const uiFontDocuments=new WeakSet<Document>();
const touchedDocuments=new WeakSet<Document>();
// Imports precede slow engine startup. Remember a touch on its loader too,
// without consuming it or making a later-mounted UI require another tap.
if(typeof document!=='undefined'){
  const doc=document;
  const remember=(event:PointerEvent):void=>{if(event.pointerType==='touch'){touchedDocuments.add(doc);doc.removeEventListener('pointerdown',remember,true);}};
  doc.addEventListener('pointerdown',remember,{capture:true,passive:true});
}
/** UI typography only: the native drawing's font and glyph bank are untouched.
 * These are the same pinned Xanh Mono bytes used by Tamanegi. */
export function applyMobileUiAppearance(element:HTMLElement):void{
  const doc=documentFor(element);
  if(!uiFontDocuments.has(doc)){
    const style=doc.createElement('style');
    const url=new URL('../../../assets/font/XanhMono-Regular.ttf',import.meta.url).href;
    style.textContent=`@font-face{font-family:"Shallot UI Xanh Mono";src:url(${JSON.stringify(url)}) format("truetype");font-style:normal;font-weight:400;font-display:swap}.shallot-mobile-ui :is(button,input,select){font:inherit}`;
    doc.head.append(style);uiFontDocuments.add(doc);
  }
  if(!element.className.split(/\s+/).includes('shallot-mobile-ui'))element.className+=' shallot-mobile-ui';
  element.style.font='14px/1.5 "Shallot UI Xanh Mono", "Xanh Mono", Consolas, monospace';
}

function documentFor(parent: HTMLElement): Document {
  const doc = parent.ownerDocument;
  if (!doc) throw new Error('Mobile controls require a parent with an ownerDocument');
  return doc;
}

function activeValue(value: boolean | (() => boolean)): boolean {
  return typeof value === 'function' ? value() : value;
}

function consume(event: Event): void {
  event.preventDefault();
  event.stopPropagation();
}

export function supportsTouchControls(view:Window|null|undefined=typeof window==='undefined'?undefined:window):boolean {
  return !!view&&(view.matchMedia?.('(any-pointer: coarse)').matches===true||(view.navigator?.maxTouchPoints??0)>0);
}
function coarsePointer(parent: HTMLElement, override: boolean | undefined): boolean {
  if (override !== undefined) return override;
  return supportsTouchControls(parent.ownerDocument?.defaultView);
}

function styleControls(root: HTMLElement, base: HTMLElement, knob: HTMLElement, actionBar: HTMLElement, options: MobileControlsOptions): void {
  applyMobileUiAppearance(root);
  Object.assign(root.style, {
    position: 'fixed',
    insetInlineStart: 'max(12px, env(safe-area-inset-left, 0px))',
    insetInlineEnd: 'max(12px, env(safe-area-inset-right, 0px))',
    insetBlockEnd: `max(${Math.max(12, options.bottomInset ?? 12)}px, env(safe-area-inset-bottom, 0px))`,
    maxWidth: 'calc(100vw - max(24px, env(safe-area-inset-left, 0px) + env(safe-area-inset-right, 0px)))',
    boxSizing: 'border-box',
    zIndex: '1000',
    display: 'flex',
    alignItems: 'flex-end',
    gap: '10px',
    pointerEvents: 'none',
    userSelect: 'none',
    WebkitUserSelect: 'none',
  });
  Object.assign(base.style, {
    position: 'relative',
    width: `${CONTROL_SIZE}px`,
    height: `${CONTROL_SIZE}px`,
    minWidth: `${TOUCH_TARGET}px`,
    minHeight: `${TOUCH_TARGET}px`,
    flex: '0 0 auto',
    borderRadius: '0',
    border: '2px solid #000',
    background: 'rgba(255,255,255,.6)',
    boxShadow: 'none',
    touchAction: 'none',
    pointerEvents: 'auto',
  });
  Object.assign(knob.style, {
    position: 'absolute',
    left: '50%',
    top: '50%',
    width: `${KNOB_SIZE}px`,
    height: `${KNOB_SIZE}px`,
    minWidth: `${TOUCH_TARGET / 2}px`,
    minHeight: `${TOUCH_TARGET / 2}px`,
    marginLeft: `${-KNOB_SIZE / 2}px`,
    marginTop: `${-KNOB_SIZE / 2}px`,
    borderRadius: '0',
    border: '2px solid #000',
    boxSizing: 'border-box',
    background: '#fff',
    transform: 'translate(0px, 0px)',
    pointerEvents: 'none',
  });
  Object.assign(actionBar.style, {
    display: 'flex',
    alignItems: 'flex-end',
    justifyContent: 'flex-end',
    flexWrap: 'wrap',
    maxWidth: 'min(52vw, 240px)',
    marginInlineStart: options.actionAlign === 'start' ? '0' : 'auto',
    gap: '8px',
    pointerEvents: 'none',
  });
}

function actionParts(actions: MobileControlsOptions['actions']): readonly [string, MobileActionSpec][] {
  if (!actions) return [];
  return Object.entries(actions).map(([name, action]) => [
    name,
    typeof action === 'function' ? { onPress: action } : action,
  ]);
}

export function mountMobileControls(options: MobileControlsOptions): MobileControlsHandle {
  const parent = options.parent ?? (() => {
    if (typeof document === 'undefined') throw new Error('Mobile controls require a DOM parent');
    return document.body;
  })();
  const doc = documentFor(parent);
  const root = doc.createElement('div');
  const base = doc.createElement('div');
  const knob = doc.createElement('div');
  const actionBar = doc.createElement('div');
  const actionButtons: Record<string, HTMLButtonElement> = {};
  const mounted = coarsePointer(parent, options.coarsePointer);
  const listeners: { target: ListenerTarget; type: string; listener: EventListener; options?:AddEventListenerOptions }[] = [];
  const actionPointers = new Map<number, {button: HTMLButtonElement; spec: MobileActionSpec}>();
  const deadzone = Math.min(0.95, Math.max(0, options.deadzone ?? DEFAULT_DEADZONE));
  let activeOverride: boolean | undefined;
  let enabled = activeValue(options.active);
  let joystickPointer: number | undefined;
  let disposed = false;
  let touched=touchedDocuments.has(doc);

  root.className = 'shallot-mobile-controls';
  root.setAttribute('aria-label', 'Mobile controls');
  root.setAttribute('data-shallot-mobile-controls', '');
  base.className = 'shallot-mobile-joystick';
  base.setAttribute('role', 'application');
  base.setAttribute('aria-label', 'Movement joystick');
  knob.className = 'shallot-mobile-joystick-knob';
  actionBar.className = 'shallot-mobile-actions';
  styleControls(root, base, knob, actionBar, options);
  base.append(knob);
  root.append(base, actionBar);
  parent.append(root);

  const listen = (target: ListenerTarget, type: string, listener: EventListener, listenerOptions?: AddEventListenerOptions): void => {
    target.addEventListener(type, listener, listenerOptions);
    listeners.push({ target, type, listener, options:listenerOptions });
  };

  const notifyZero = (): void => options.onMove(0, 0);

  const releaseJoystickCapture = (pointerId: number): void => {
    if ('releasePointerCapture' in base && base.hasPointerCapture?.(pointerId)) base.releasePointerCapture(pointerId);
  };

  const stopJoystick = (notify = true): void => {
    const pointerId = joystickPointer;
    joystickPointer = undefined;
    if (pointerId !== undefined) releaseJoystickCapture(pointerId);
    knob.style.transform = 'translate(0px, 0px)';
    if (notify) notifyZero();
  };

  const stopActions = (): void => {
    for (const [pointerId, activeAction] of actionPointers) {
      const {button, spec} = activeAction;
      actionPointers.delete(pointerId);
      if ('releasePointerCapture' in button && button.hasPointerCapture?.(pointerId)) button.releasePointerCapture(pointerId);
      spec.onCancel?.();
    }
  };

  const reset = (): void => {
    if (disposed) return;
    stopJoystick(true);
    stopActions();
  };

  const refresh = (): void => {
    if (disposed) return;
    enabled = activeOverride ?? activeValue(options.active);
    root.hidden = !mounted || !enabled || !touched;
    root.style.display=root.hidden?'none':'flex';
    root.setAttribute('aria-hidden', String(root.hidden));
    if (!enabled) reset();
  };

  const firstTouch=(event:Event):void=>{
    if(!touched&&(event as PointerEvent).pointerType==='touch'){touched=true;touchedDocuments.add(doc);refresh();}
  };
  // Observe only: the same first gesture must still reach native look/fire.
  listen(doc,'pointerdown',firstTouch,{capture:true,passive:true});

  const eventActive = (): boolean => {
    if (!enabled||!mounted||!touched) return false;
    enabled = activeOverride ?? activeValue(options.active);
    if (!enabled) reset();
    return enabled;
  };

  const moveFrom = (event: PointerLike): void => {
    const rect = base.getBoundingClientRect();
    const radius = Math.max(1, Math.min(rect.width, rect.height) / 2);
    let x = (event.clientX - (rect.left + rect.width / 2)) / radius;
    let y = (event.clientY - (rect.top + rect.height / 2)) / radius;
    const length = Math.hypot(x, y);
    if (length <= deadzone) {
      x = 0;
      y = 0;
    } else {
      const scaled = (Math.min(1, length) - deadzone) / (1 - deadzone);
      const direction = Math.max(length, 1e-9);
      x = (x / direction) * scaled;
      y = (y / direction) * scaled;
    }
    knob.style.transform = `translate(${x * (CONTROL_SIZE - KNOB_SIZE) / 2}px, ${y * (CONTROL_SIZE - KNOB_SIZE) / 2}px)`;
    options.onMove(x, y);
  };

  const joystickDown = (event: Event): void => {
    consume(event);
    firstTouch(event);
    if (!eventActive()) return;
    const pointer = event as PointerLike;
    if (joystickPointer !== undefined) return;
    joystickPointer = pointer.pointerId;
    base.setPointerCapture(pointer.pointerId);
    moveFrom(pointer);
  };
  const joystickMove = (event: Event): void => {
    consume(event);
    const pointer = event as PointerLike;
    if (pointer.pointerId !== joystickPointer || !eventActive()) return;
    moveFrom(pointer);
  };
  const joystickEnd = (event: Event): void => {
    consume(event);
    const pointer = event as PointerLike;
    if (pointer.pointerId !== joystickPointer) return;
    stopJoystick(true);
  };

  listen(base, 'pointerdown', joystickDown);
  listen(base, 'pointermove', joystickMove);
  listen(base, 'pointerup', joystickEnd);
  listen(base, 'pointercancel', joystickEnd);
  listen(base, 'lostpointercapture', joystickEnd);

  for (const [name, spec] of actionParts(options.actions)) {
    const button = doc.createElement('button');
    button.type = 'button';
    button.className = 'shallot-mobile-action';
    button.textContent = spec.label ?? name;
    button.setAttribute('aria-label', spec.ariaLabel ?? spec.label ?? name);
    button.setAttribute('data-action', name);
    Object.assign(button.style, {
      minWidth: `${TOUCH_TARGET}px`,
      minHeight: `${TOUCH_TARGET}px`,
      padding: '8px',
      border: '2px solid #000',
      borderRadius: '0',
      background: '#fff',
      color: '#000',
      boxShadow: 'none',
      font: 'inherit',
      fontSize: '16px',
      pointerEvents:'auto',
      touchAction: 'none',
      userSelect: 'none',
    });
    actionButtons[name] = button;
    actionBar.append(button);
    const actionDown = (event: Event): void => {
      consume(event);
      firstTouch(event);
      if (!eventActive()) return;
      const pointer = event as PointerLike;
      if (actionPointers.has(pointer.pointerId)) return;
      actionPointers.set(pointer.pointerId, {button, spec});
      button.setPointerCapture(pointer.pointerId);
      spec.onPress(pointer);
    };
    const actionEnd = (event: Event): void => {
      consume(event);
      const pointer = event as PointerLike;
      const activeAction = actionPointers.get(pointer.pointerId);
      if (!activeAction || activeAction.button !== button) return;
      actionPointers.delete(pointer.pointerId);
      if (event.type === 'pointerup' && eventActive()) spec.onRelease?.(pointer);
      else activeAction.spec.onCancel?.();
    };
    listen(button, 'pointerdown', actionDown);
    listen(button, 'pointerup', actionEnd);
    listen(button, 'pointercancel', actionEnd);
    listen(button, 'lostpointercapture', actionEnd);
    listen(button, 'click', consume);
  }

  const resetFromWindow = (): void => reset();
  const view = doc.defaultView;
  if (view) listen(view, 'blur', resetFromWindow);
  listen(doc, 'visibilitychange', () => {
    if (doc.visibilityState === 'hidden') reset();
  });

  const setActive = (value: boolean): void => {
    activeOverride = value;
    refresh();
  };

  const dispose = (): void => {
    if (disposed) return;
    reset();
    disposed = true;
    for (const { target, type, listener, options } of listeners) target.removeEventListener(type, listener,options);
    listeners.length = 0;
    root.remove();
  };

  refresh();
  return { element: root, joystick: base, knob, actionButtons, mounted, refresh, reset, setActive, dispose };
}
