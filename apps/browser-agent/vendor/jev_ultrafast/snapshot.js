// Adapted from browser-use/jev-ultrafast @1231850a, MIT License.
// Adds generic visible cursor/onclick/label controls; no site-specific selectors.
(() => {
  if (!document.body) return null;
  const cache = window.__jevFast ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  // Open shadow roots are part of the page a person sees: consent dialogs and web components render
  // there. browser-use buildDomTree.js (0.3.2) walks node.shadowRoot the same way.
  const hosts=[];
  for (let i=-1;i<hosts.length;i++)
    for (const e of (i<0 ? document : hosts[i].shadowRoot).querySelectorAll('*')) if (e.shadowRoot) hosts.push(e);
  cache.hosts=hosts;
  const roots=()=>[document,...cache.hosts.filter(h=>h.isConnected && h.shadowRoot).map(h=>h.shadowRoot)];
  // What a pointer at (x,y) reaches, looking inside open shadow roots, and containment across shadow
  // boundaries. browser-use buildDomTree.js isTopElement asks the element's own shadow root for
  // elementFromPoint and walks up from the result. Kept on the cache for the click-time check (run.py).
  cache.deepAt=(x,y)=>{
    let e=document.elementFromPoint(x,y);
    while (e?.shadowRoot) {
      const inner=e.shadowRoot.elementFromPoint(x,y);
      if (!inner || inner===e) break;
      e=inner;
    }
    return e;
  };
  cache.within=(outer,node)=>{
    for (let n=node;n;n=n.assignedSlot||n.parentNode||n.host) if (n===outer) return true;
    return false;
  };
  let reading=true;
  const styleMemo=new WeakMap(),nameMemo=new WeakMap(),roleMemo=new WeakMap();
  const style=e=>{
    if (!reading) return getComputedStyle(e);
    if (!styleMemo.has(e)) styleMemo.set(e,getComputedStyle(e));
    return styleMemo.get(e);
  };
  const safe = e => !['password','file','hidden'].includes(e.type);
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const name = (e,seen) => {
    if (seen) return rawName(e,seen);
    if (reading && nameMemo.has(e)) return nameMemo.get(e);
    const result=rawName(e,new Set());
    if (reading && e) nameMemo.set(e,result);
    return result;
  };
  const rawName = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(e.getRootNode().getElementById?.(id)||document.getElementById(id),seen)).filter(Boolean).join(' ');
    return referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && n.getAttribute('aria-hidden')!=='true' && visible(n) ? name(n,seen) : '').join(' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
  };
  const activates=e=> {
    if (['onclick','onmousedown','onmouseup','onpointerdown','onpointerup'].some(k=>typeof e[k]==='function')) return true;
    const key=Object.keys(e).find(k=>k.startsWith('__reactProps$'));
    const props=key && e[key];
    return !!props && ['onClick','onMouseDown','onMouseUp','onPointerDown','onPointerUp'].some(k=>typeof props[k]==='function');
  };
  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const selector='a,button,input,textarea,select,summary,[contenteditable="true"],'+
    roles.map(role=>'[role="'+role+'"]').join(',')+',div,span,li,label,i,svg,td,strong,b,em,[onclick],[tabindex]';
  const role = e => {
    if (reading && roleMemo.has(e)) return roleMemo.get(e);
    const result=rawRole(e);
    if (reading) roleMemo.set(e,result);
    return result;
  };
  const rawRole = e => {
    const explicit=e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A' && e.hasAttribute('href')) return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='LABEL' && ['checkbox','radio'].includes(e.control?.type)) return e.control.type;
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      // Readonly inputs inside an actual select component are dropdowns, not
      // editable text fields. Keep ordinary readonly text as a textbox.
      if (e.readOnly) {
        if (e.getAttribute('aria-haspopup')==='listbox' || e.closest('[role="combobox"],.el-select')) return 'combobox';
        for (let parent=e.parentElement,depth=0;parent && depth<3;parent=parent.parentElement,depth++) {
          if (parent.__vue__?.$options?.name==='ElSelect') return 'combobox';
        }
      }
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    // Generic custom controls used by Chinese consumer sites. Only visible DOM
    // nodes become candidates; Jev still chooses the operation and node index.
    if (style(e).cursor==='pointer' || activates(e) || e.hasAttribute('onclick') || e.tagName==='LABEL') {
      const label=name(e).trim();
      const r=e.getBoundingClientRect();
      const backdrop=!label && /mask|backdrop|overlay/i.test(e.getAttribute('class')||'') && typeof e.onclick==='function';
      if (label.length>350 || (!backdrop && r.height>180)) return null;
      if (!label && !backdrop && (r.width>100 || r.height>100 || !(e.getAttribute('class')||e.getAttribute('title')))) return null;
      return 'button';
    }
    return null;
  };
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    roots().flatMap(root=>[...root.querySelectorAll('input,textarea,select')]).filter(safe)
      .map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),e.value??null,e.checked??e.control?.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled')||!!e.control?.disabled,e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      e.getAttribute('href'),scope?.innerText?.slice(0,6000)||''];
  };
  const actions=[];
  // Composed-tree order: a host's shadow content follows the host, so a consent dialog keeps its place
  // in the element list (which is cut at 250). Pages without open shadow roots take the original query.
  const candidates=()=>{
    if (!cache.hosts.length) return document.querySelectorAll(selector);
    const found=[];
    const walk=root=>{
      for (const e of root.querySelectorAll('*')) {
        if (e.matches(selector)) found.push(e);
        if (e.shadowRoot) walk(e.shadowRoot);
      }
    };
    walk(document);
    return found;
  };
  for (const e of candidates()) {
    if (!safe(e) || !visible(e) || e.matches(':disabled') || e.control?.disabled || e.closest('[aria-disabled="true"]')) continue;
    const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, rname=role(e);
    if (!rname || r.width<=0 || r.height<=0 || x<0 || y<0 || x>=innerWidth || y>=innerHeight) continue;
    // A covered control cannot execute; omit it until the covering menu closes.
    const hittable=[.5,.25,.75,.1,.9].some(f=>{
      const px=r.x+r.width*f,py=r.y+r.height*.5;
      return px>=0 && px<innerWidth && py>=0 && py<innerHeight && cache.within(e,cache.deepAt(px,py));
    });
    if (!hittable) continue;
    // Nested labels and icons often repeat the same click handler. Keep the
    // outer observed control so labels reach their actual handler.
    const ownName=name(e).trim();
    let duplicate=false;
    for (let parent=e.parentElement,depth=0;parent && depth<3;parent=parent.parentElement,depth++) {
      const pr=role(parent),pn=name(parent).trim();
      if (pr && visible(parent) && ((ownName && ownName===pn) || (!ownName && pn))) {
        duplicate=true;break;
      }
    }
    if (duplicate) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height},
      opens_new_tab:(e.closest('a')?.target==='_blank' || !!e.querySelector('a[target="_blank"]'))};
    if (!name(e)) {
      if (['INPUT','TEXTAREA'].includes(e.tagName)) {
        const scope=e.parentElement?.parentElement || e.parentElement;
        const siblings=[...(scope?.querySelectorAll('input,textarea')||[])].filter(n=>visible(n)&&safe(n));
        base.label=(scope?.innerText?.trim().slice(0,100)||'Input field')+
          ` · input field ${siblings.indexOf(e)+1} of ${siblings.length}`;
      } else if (/mask|backdrop|overlay/i.test(e.getAttribute('class')||'') && typeof e.onclick==='function') {
        base.label=`Blank backdrop that closes the current overlay ${e.getAttribute('class')}`;
      } else base.label=`Unlabeled icon ${e.tagName}.${e.getAttribute('class')||''}`;
    }
    base.parent_class=typeof e.parentElement?.className==='string'?e.parentElement.className:'';
    base.own_class=typeof e.className==='string'?e.className:'';
    base.dom_state=[e.className,
      ...[...(e.querySelectorAll('i,[role="checkbox"],[role="radio"]')||[])]
        .slice(0,4).map(n=>n.className)].filter(x=>typeof x==='string').join(' | ').slice(0,120);
    base.nearby_text=(e.parentElement?.innerText||'').trim().slice(0,120);
    base.position={x:Math.round(r.x),y:Math.round(r.y)};
    if (['INPUT','TEXTAREA'].includes(e.tagName) && base.nearby_text &&
        !base.label.includes(base.nearby_text.slice(0,80))) {
      base.label+=' · '+base.nearby_text.slice(0,100);
    }
    if (['INPUT','TEXTAREA'].includes(e.tagName)) {
      // Form rows often label an input using the visible value of a neighboring
      // readonly dropdown, which innerText does not include. Read that context
      // without selecting a field or prescribing a value.
      let rowValues=[];
      for (let parent=e.parentElement,depth=0;parent && depth<5;parent=parent.parentElement,depth++) {
        if (parent.getBoundingClientRect().height>Math.max(90,r.height*3)) break;
        const fields=[...parent.querySelectorAll('input,textarea,select')].filter(n=>safe(n)&&visible(n));
        if (fields.length>8) break;
        const values=fields.filter(n=>n!==e && (n.readOnly || n.tagName==='SELECT'))
          .map(n=>n.tagName==='SELECT' ? [...n.selectedOptions].map(o=>o.label).join(' ') : n.value).filter(Boolean);
        if (values.length>rowValues.length) rowValues=values;
      }
      base.row_context=[...new Set(rowValues)].join(' / ').slice(0,140);
      if (base.row_context && !base.label.includes(base.row_context)) base.label+=' · '+base.row_context;
      base.input_constraints={type:e.type,readonly:!!e.readOnly};
      for (const key of ['pattern','min','max','step','inputmode','maxlength']) {
        const value=e.getAttribute(key); if (value!==null) base.input_constraints[key]=value;
      }
      // Expose only declared primitive format metadata from the visible input's
      // component. Do not read application stores or invoke component handlers.
      for (let parent=e.parentElement,depth=0;parent && depth<3;parent=parent.parentElement,depth++) {
        const props=parent.__vue__?.$props;
        if (typeof props?.format==='string' && props.format) {
          base.input_constraints.format=props.format;
          if (typeof props.valueFormat==='string') base.input_constraints.value_format=props.valueFormat;
          if (typeof props.type==='string') base.input_constraints.component_type=props.type;
          base.label+=' · format '+props.format;
          break;
        }
      }
    }
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=value;
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);
    else if (e.tagName==='LABEL' && ['checkbox','radio'].includes(e.control?.type)) base.checked=String(e.control.checked);
    if (e.tagName==='SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        actions.push({...base,kind:'select',value:o.value,
          current_value:[...e.selectedOptions].map(o=>o.label).join(', '),label:base.label+' → '+o.label});
    } else {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      // Filling already focuses the field with a real pointer event. Avoid a
      // redundant Open action for ordinary editable fields; editable comboboxes
      // retain it because opening their option list can be independently useful.
      if (editable && rname==='combobox') actions.push({...base,kind:'click',value,label:'Open '+base.label});
    }
  }
  // Panels can scroll independently of the document (comments, lists, dialogs).
  for (const e of roots().flatMap(root=>[...root.querySelectorAll('div,section,main,aside,ul')])) {
    const r=e.getBoundingClientRect(),computed=style(e);
    if (!visible(e) || !/auto|scroll/.test(computed.overflowY) || e.scrollHeight-e.clientHeight<80 || r.height<120 || r.width<140) continue;
    const x=Math.max(0,r.x)+Math.min(r.width,innerWidth-Math.max(0,r.x))/2;
    const y=Math.max(0,r.y)+Math.min(r.height,innerHeight-Math.max(0,r.y))/2;
    if (x<0||y<0||x>=innerWidth||y>=innerHeight||!cache.within(e,cache.deepAt(x,y))) continue;
    const node=identity(e),label=(e.innerText||'').trim().slice(0,90);
    const base={node,kind:'scroll_panel',role:'region',value:String(e.scrollTop),rect:{x:r.x,y:r.y,w:r.width,h:r.height}};
    if (e.scrollTop+e.clientHeight<e.scrollHeight-3) actions.push({...base,label:'Scroll down within the region: '+label,delta:480});
    if (e.scrollTop>0) actions.push({...base,label:'Scroll up within the region: '+label,delta:-480});
  }
  // Text is readable where nothing opaque is painted over it. As in browser-use paint_order.py, an
  // element with a transparent background or opacity below 0.8 does not hide what lies under it (a card's
  // transparent link overlay, for one). Clickable elements keep the strict check above: a click has to
  // land on the element itself.
  const opaque=e=>style(e).backgroundColor!=='rgba(0, 0, 0, 0)' && parseFloat(style(e).opacity)>=0.8;
  const stackAt=(x,y,root=document)=>{
    const found=[];
    for (const e of root.elementsFromPoint(x,y)) {
      if (root!==document && !root.contains(e)) continue;  // a shadow root also reports the page beneath it
      if (e.shadowRoot && e.shadowRoot!==root) found.push(...stackAt(x,y,e.shadowRoot));
      found.push(e);
    }
    return found;
  };
  const readable=(parent,x,y)=>{
    if (cache.within(parent,cache.deepAt(x,y))) return true;
    for (const e of stackAt(x,y)) {
      if (cache.within(parent,e)) return true;
      if (opaque(e)) return false;
    }
    return false;
  };
  // Visible text in document order, open shadow roots included; from <html>, because consent managers attach
  // their host next to <body> (<head> content is never visible). An image's description counts as text
  // (browser-use serializer.py add_image_context reads img alt, title and aria-label), except inside a
  // link or button, whose label above already carries it.
  const nodes=function*(root){
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT|NodeFilter.SHOW_TEXT);
    for (let node=walker.nextNode();node;node=walker.nextNode()) {
      yield node;
      if (node.shadowRoot) yield* nodes(node.shadowRoot);
    }
  };
  const words=[], range=document.createRange(); let length=0;
  for (const node of nodes(document.documentElement)) {
    if (length>=6000) break;
    if (node.nodeType===1) {
      if (node.tagName!=='IMG' || node.closest('a,button,[role="button"],[role="link"]')) continue;
      const value=(node.getAttribute('alt')||node.getAttribute('title')||node.getAttribute('aria-label')||'')
        .replace(/\s+/g,' ').trim().slice(0,500);
      const r=node.getBoundingClientRect();
      if (!value || !visible(node) || r.width<=0 || r.height<=0 || r.bottom<=0 || r.top>=innerHeight ||
          r.right<=0 || r.left>=innerWidth) continue;
      const x=(Math.max(0,r.left)+Math.min(innerWidth-1,r.right))/2;
      const y=(Math.max(0,r.top)+Math.min(innerHeight-1,r.bottom))/2;
      if (readable(node,x,y)) {words.push(value); length+=value.length;}
      continue;
    }
    const value=node.textContent.trim(), parent=node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
    range.selectNodeContents(node); const r=range.getBoundingClientRect();
    const uncovered=[...range.getClientRects()].some(box=>{
      if (box.bottom<=0||box.top>=innerHeight||box.right<=0||box.left>=innerWidth) return false;
      const x=(Math.max(0,box.left)+Math.min(innerWidth-1,box.right))/2;
      const y=(Math.max(0,box.top)+Math.min(innerHeight-1,box.bottom))/2;
      return readable(parent,x,y);
    });
    if (uncovered && r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
      let clipped=value;
      if (value.length>80) {
        // A line-clamped comment may have a single text node containing hidden
        // prose. Read character ranges so a visible prefix cannot expose it.
        let result='',offset=0,gap=false;
        for (const char of node.textContent) {
          range.setStart(node,offset); offset+=char.length; range.setEnd(node,offset);
          const box=range.getBoundingClientRect();
          const x=(Math.max(0,box.left)+Math.min(innerWidth-1,box.right))/2;
          const y=(Math.max(0,box.top)+Math.min(innerHeight-1,box.bottom))/2;
          const shown=box.width>0 && box.height>0 && box.bottom>0 && box.top<innerHeight && box.right>0 && box.left<innerWidth && readable(parent,x,y);
          if (shown) { if(gap && result) result+='…';result+=char;gap=false; }
          else if (char.trim()) gap=true;
        }
        if(gap && result) result+='…';
        clipped=result.trim();
      }
      if(clipped) {words.push(clipped); length+=clipped.length;}
    }
  }
  const text=words.join('\n').slice(0,6000), height=document.documentElement.scrollHeight;
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  // Compare meaning and identity. Geometry is always resolved and hit-tested just before input.
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6]];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:560});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-560});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  reading=false; // Freshness guards must always recompute live semantics.
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
})()
