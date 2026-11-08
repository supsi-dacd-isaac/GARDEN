/* Native annual trace viewer. Script payloads permit direct file:// use. */
"use strict";
(() => {
  const M = window.ANNUAL_TRACE_MANIFEST;
  const $ = id => document.getElementById(id);
  const plot = $("plot"), DAY = 86400000;
  const channels = ["Tin", "Qroom", "Pel_SH"];
  const titles = ["Indoor temperature", "Delivered room heat", "HP electric power"];
  const cache = new Map(), pending = new Map();
  let serial = 0, current = null, windowRange = null, busy = false, plotted = false, eventIndex = -1;
  let rendering = Promise.resolve();
  const iso = n => new Date(n).toISOString().slice(0, 23);
  const parseTime = v => typeof v === "number" ? v : Date.parse(v.replace(" ", "T") + (/Z$|[+-]\d\d:\d\d$/.test(v) ? "" : "Z"));

  window.registerAnnualTrace = (key, payload) => {
    cache.set(key, payload);
    if (pending.has(key)) { pending.get(key).resolve(payload); pending.delete(key); }
    // Bound retained file data when users browse many buildings.
    while (cache.size > 8) cache.delete(cache.keys().next().value);
  };
  function load(key, file) {
    if (cache.has(key)) return Promise.resolve(cache.get(key));
    if (pending.has(key)) return pending.get(key).promise;
    let resolve, reject;
    const promise = new Promise((a, b) => { resolve = a; reject = b; });
    pending.set(key, {promise, resolve, reject});
    const script = document.createElement("script");
    script.src = file;
    script.onload = () => {
      if (pending.has(key)) { pending.get(key).reject(new Error(`Invalid trace payload: ${file}`)); pending.delete(key); }
      script.remove();
    };
    script.onerror = () => { pending.delete(key); reject(new Error(`Cannot load ${file}. Keep the data folder beside index.html.`)); script.remove(); };
    document.head.appendChild(script);
    return promise;
  }
  function unpack(s) {
    const raw = atob(s), bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function policyName() { return $("dataset").value === "tariff" ? `tariff_${$("policy").value}_v2` : `${$("policy").value}_excitation_v2`; }
  function fail(error) {
    $("loading").hidden = true;
    $("status").textContent = error.message || String(error);
    $("status").dataset.error = "true";
    console.error(error);
  }
  function bounds() { return [current.start_ms, current.start_ms + current.count * current.step_ms]; }
  function clampRange(range) {
    const [lo, hi] = bounds();
    return [Math.max(lo, Math.min(range[0], hi - current.step_ms)), Math.min(hi, Math.max(range[1], lo + current.step_ms))];
  }
  function fitY(channel, range) {
    const first = Math.max(0, Math.floor((range[0] - current.start_ms) / current.step_ms));
    const last = Math.min(current.count, Math.ceil((range[1] - current.start_ms) / current.step_ms) + 1);
    const arrays = [current.truth[channel], current.mean[channel], current.lower[channel], current.upper[channel]];
    if (channel === "Tin") arrays.push(current.setpoint);
    let lo = channel === "Tin" ? Infinity : 0, hi = -Infinity;
    for (const a of arrays) for (let i = first; i < last; i++) { lo = Math.min(lo, a[i]); hi = Math.max(hi, a[i]); }
    const pad = Math.max((hi - lo) * 0.07, channel === "Tin" ? .12 : .25);
    return [lo - pad, hi + pad];
  }
  function shapes(range) {
    return current.force_off_intervals.filter(([a, b]) => b > range[0] && a < range[1]).map(([a, b]) => ({
      type:"rect", xref:"x3", yref:"y3 domain", x0:iso(a), x1:iso(b), y0:0, y1:1,
      fillcolor:"rgba(230,159,45,0.21)", line:{width:0}, layer:"below",
    }));
  }
  function updateControls() {
    $("from").value = iso(windowRange[0]).slice(0, 10);
    $("to").value = iso(windowRange[1] - 1).slice(0, 10);
    const count = current.force_off_intervals.length;
    $("previous-event").disabled = $("next-event").disabled = !count;
    $("force-text").textContent = count ? `${count} force-off periods · ${current.force_off_hours.toLocaleString()} h/year` : "No force-off requests in this policy.";
    $("force-summary").style.opacity = count ? "1" : ".6";
  }
  function reflectURL() {
    const params = new URLSearchParams({policy:$("policy").value, dataset:$("dataset").value, model:$("model").value, egid:$("egid").value});
    if (windowRange) { params.set("start", iso(windowRange[0])); params.set("end", iso(windowRange[1])); }
    history.replaceState(null, "", "#" + params.toString());
  }
  async function setRange(range) {
    if (!current || busy) return;
    windowRange = clampRange(range);
    const update = {shapes:shapes(windowRange)};
    for (let i = 0; i < 3; i++) {
      const suffix = i ? String(i + 1) : "";
      update[`xaxis${suffix}.range`] = windowRange.map(iso);
      update[`xaxis${suffix}.autorange`] = false;
      update[`yaxis${suffix}.range`] = fitY(channels[i], windowRange);
    }
    busy = true;
    try { await Plotly.relayout(plot, update); updateControls(); reflectURL(); }
    finally { busy = false; }
  }
  function traces() {
    const data = [];
    channels.forEach((channel, i) => {
      const axis = i ? String(i + 1) : "";
      const shared = {type:"scatter", mode:"lines", x:current.x, xaxis:`x${axis}`, yaxis:`y${axis}`};
      data.push({...shared, y:current.lower[channel], name:"95% predictive band", legendgroup:"band", showlegend:false, line:{width:0}, hoverinfo:"skip"});
      data.push({...shared, y:current.upper[channel], name:"95% predictive band", legendgroup:"band", showlegend:i===0, line:{width:0}, fill:"tonexty", fillcolor:"rgba(211,47,47,0.15)", hoverinfo:"skip"});
      data.push({...shared, y:current.truth[channel], name:"EnergyPlus", legendgroup:"truth", showlegend:i===0, line:{color:"#1769aa",width:1.5}, hovertemplate:"%{y:.3f}<extra>EnergyPlus</extra>"});
      data.push({...shared, y:current.mean[channel], name:"Emulator mean", legendgroup:"mean", showlegend:i===0, line:{color:"#d32f2f",width:1.6}, hovertemplate:"%{y:.3f}<extra>Emulator mean</extra>"});
      if (i === 0) data.push({...shared, y:current.setpoint, name:"Heating setpoint", legendgroup:"setpoint", line:{color:"#697582",width:1,dash:"dot",shape:"hv"}, hovertemplate:"%{y:.2f} °C<extra>Setpoint</extra>"});
    });
    return data;
  }
  async function render(ticket) {
    if (ticket !== serial) return;
    const domains = [[.72, 1], [.36, .64], [0, .28]];
    const layout = {
      autosize:true, height:900, margin:{l:76,r:28,t:100,b:65}, paper_bgcolor:"white", plot_bgcolor:"white",
      font:{family:"system-ui, -apple-system, sans-serif",color:"#243f54",size:12},
      hovermode:"x unified", hoverlabel:{namelength:-1}, dragmode:"zoom", shapes:shapes(windowRange),
      legend:{orientation:"h",x:0,y:1.12,groupclick:"togglegroup"},
      annotations:titles.map((text, i) => ({text, x:0,y:domains[i][1]+.012,xref:"paper",yref:"paper",xanchor:"left",yanchor:"bottom",showarrow:false,font:{size:16,color:"#182f42"}})),
    };
    for (let i = 0; i < 3; i++) {
      const suffix = i ? String(i + 1) : "";
      layout[`xaxis${suffix}`] = {type:"date",anchor:`y${suffix}`,domain:[0,1],range:windowRange.map(iso),autorange:false,
        matches:i===2?undefined:"x3",showticklabels:true,gridcolor:"#edf1f5",showspikes:true,spikemode:"across",spikesnap:"cursor",
        title:i===2?{text:"2023 · local standard time"}:undefined};
      layout[`yaxis${suffix}`] = {anchor:`x${suffix}`,domain:domains[i],range:fitY(channels[i],windowRange),fixedrange:true,
        title:{text:i===0?"°C":"W/m²",standoff:12},gridcolor:"#e8eef3",zerolinecolor:"#d6e0e8"};
    }
    busy = true;
    try {
      await Plotly.react(plot, traces(), layout, {responsive:true,displaylogo:false,scrollZoom:false,
        doubleClick:false,modeBarButtonsToRemove:["select2d","lasso2d","autoScale2d","resetScale2d"],
        toImageButtonOptions:{format:"png",filename:`${current.policy}_${current.model}_${current.egid}`,scale:2}});
      if (!plotted) {
        plot.on("plotly_relayout", event => {
          if (busy || !current) return;
          for (const axis of ["xaxis","xaxis2","xaxis3"]) {
            const range = event[`${axis}.range`] || (event[`${axis}.range[0]`] !== undefined ? [event[`${axis}.range[0]`],event[`${axis}.range[1]`]] : null);
            if (range) { setRange(range.map(parseTime)).catch(fail); return; }
            if (event[`${axis}.autorange`]) { setRange(bounds()).catch(fail); return; }
          }
        });
        plotted = true;
      }
    } finally { busy = false; }
    if (ticket !== serial) return;
    updateControls(); reflectURL();
    $("status").textContent = `${M.models[current.model]} · EGID ${current.egid} · ${current.count.toLocaleString()} annual samples`;
    $("status").dataset.error = "false";
    $("source").href = M.entries[current.key].source;
    $("loading").hidden = true;
  }
  async function choose() {
    const ticket = ++serial;
    $("loading").hidden = false;
    $("status").textContent = "Loading saved annual traces…";
    const policy = policyName(), model = $("model").value, egid = $("egid").value;
    const key = `${policy}/${model}/${egid}`, commonKey = `${policy}/${egid}`;
    try {
      if (!M.entries[key]) throw new Error(`No saved trace for ${key}`);
      const [observed, forecast] = await Promise.all([load(commonKey,M.common[commonKey]),load(key,M.entries[key].file)]);
      if (ticket !== serial) return;
      const next = {...observed,policy,model,egid,key};
      next.setpoint = unpack(observed.setpoint);
      for (const group of ["truth","mean","lower","upper"]) {
        next[group] = {};
        for (const channel of channels) next[group][channel] = unpack((group === "truth" ? observed : forecast)[group][channel]);
      }
      next.x = Array.from({length:next.count}, (_,i) => iso(next.start_ms+i*next.step_ms));
      // Serialize Plotly updates so rapid selector changes cannot leave stale traces.
      rendering = rendering.catch(() => {}).then(async () => {
        if (ticket !== serial) return;
        current = next; eventIndex = -1;
        windowRange = windowRange ? clampRange(windowRange) : bounds();
        await render(ticket);
      });
      await rendering;
    } catch (error) { if (ticket === serial) fail(error); }
  }
  function span(days) {
    if (!current || busy) return;
    const [lo, hi] = bounds(), length = days * DAY;
    const start = Math.max(lo, Math.min(windowRange[0], hi - length));
    setRange([start,Math.min(start+length,hi)]).catch(fail);
  }
  function shift(direction) {
    if (!current || busy) return;
    const [lo,hi] = bounds(), width = windowRange[1]-windowRange[0];
    const start = Math.max(lo,Math.min(hi-width,windowRange[0]+direction*width));
    setRange([start,start+width]).catch(fail);
  }
  function event(direction) {
    if (!current || busy || !current.force_off_intervals.length) return;
    const events = current.force_off_intervals;
    if (eventIndex < 0) {
      eventIndex = events.findIndex(([start]) => start >= windowRange[0]);
      if (eventIndex < 0) eventIndex = direction > 0 ? 0 : events.length-1;
      else if (direction < 0) eventIndex = Math.max(0,eventIndex-1);
    } else eventIndex = (eventIndex+direction+events.length)%events.length;
    const [a,b] = events[eventIndex];
    setRange([a-12*3600000,b+18*3600000]).catch(fail);
  }
  for (const [value,label] of Object.entries(M.models)) $("model").add(new Option(label,value));
  for (const egid of M.egids) $("egid").add(new Option(egid,egid));
  $("egid").value = M.default_egid;
  const initial = new URLSearchParams(location.hash.slice(1));
  for (const id of ["policy","dataset","model","egid"]) {
    const value = initial.get(id);
    if (value && Array.from($(id).options).some(x => x.value === value)) $(id).value = value;
    $(id).addEventListener("change",choose);
  }
  if (initial.has("start") && initial.has("end")) {
    const candidate = [parseTime(initial.get("start")),parseTime(initial.get("end"))];
    if (candidate.every(Number.isFinite) && candidate[0] < candidate[1]) windowRange = candidate;
  }
  $("full").onclick = () => current && setRange(bounds()).catch(fail);
  $("week").onclick = () => span(7); $("month").onclick = () => span(30);
  $("previous").onclick = () => shift(-1); $("next").onclick = () => shift(1);
  $("previous-event").onclick = () => event(-1); $("next-event").onclick = () => event(1);
  $("apply").onclick = () => {
    const range = [Date.parse($("from").value+"T00:00:00Z"),Date.parse($("to").value+"T00:00:00Z")+DAY];
    if (!range.every(Number.isFinite) || range[0] >= range[1]) { $("status").textContent = "Choose a valid start and end date."; return; }
    setRange(range).catch(fail);
  };
  $("provenance").textContent = `Run: ${M.run}. Selected data: v9-final, test fold. The original plots are linked above for each selection.`;
  choose();
})();
