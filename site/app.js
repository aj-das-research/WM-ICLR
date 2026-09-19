"use strict";
const byId = id => document.getElementById(id);
let results, showcase, demoBenchmark="pusht", resultsBenchmark="pusht", playbackTimer;
const signed = (n, digits=2) => `${n>=0?"+":"−"}${Math.abs(n).toFixed(digits)}`;
const names={pusht:"PushT",reacher:"Reacher",droid:"DROID"};
function reading(target, comparison, label) {
  const gain=comparison.relative_reduction_percent, [lo,hi]=comparison.ci95;
  target.replaceChildren();
  const number=document.createElement("strong");number.className=gain>0?"positive":"negative";
  number.textContent=`${Math.abs(gain).toFixed(2)}% ${gain>0?"lower":"higher"} error`;
  const name=document.createElement("span");name.textContent=`vs ${label}`;
  const interval=document.createElement("p");interval.className="fineprint";
  interval.textContent=`95% CI, MSE difference: [${signed(lo,6)}, ${signed(hi,6)}]. ${hi<0||lo>0?"Interval excludes zero.":"Interval includes zero."}`;
  target.append(number,name,interval);
}
function row(target,label,mse,sd,ours=false) {
  const tr=document.createElement("tr");if(ours)tr.className="ours-row";
  const name=document.createElement("th");name.scope="row";name.textContent=label;
  const value=document.createElement("td");value.textContent=mse.toFixed(6);
  const spread=document.createElement("td");spread.textContent=sd===null?"—":sd.toFixed(6);
  tr.append(name,value,spread);target.append(tr);
}
function renderResults() {
  if(!results)return;
  const camera=byId("result-camera").value,horizon=Number(byId("result-horizon").value);
  const p=results.populations.find(p=>p.camera===camera&&p.horizon===horizon);
  byId("result-rows").replaceChildren();
  p.rows.forEach(r=>row(byId("result-rows"),r.mode==="factorized"?"Historical context model":r.name,r.mse,r.seed_sd,r.mode==="factorized"));
  byId("population-count").textContent=`${p.episodes} held-out episodes · ${p.sessions} sessions`;
  reading(byId("framewise-gain"),p.comparisons.framewise,"Framewise");
  reading(byId("persistence-gain"),p.comparisons.persistence,"persistence");
  byId("population-note").textContent=horizon===5?"Five action blocks is the trained horizon. Camera 1 is the prespecified primary comparison.":"Ten blocks extrapolates beyond the five-block training horizon. Its eligible episode population is slightly smaller.";
}
function renderSimResults() {
  if(!showcase||resultsBenchmark==="droid")return;
  const data=showcase.benchmarks[resultsBenchmark],split=byId("sim-split").value;
  byId("sim-result-rows").replaceChildren();
  data.forecast_rows.filter(r=>r.split===split).forEach(r=>row(byId("sim-result-rows"),r.mode==="factorized"?"Historical context model":r.label,r.value,r.uncertainty.value,r.mode==="factorized"));
  const comp=data.comparisons.find(r=>r.split===split&&r.reference==="framewise");
  const gain=-comp.relative_change_percent;
  const strong=document.createElement("strong");strong.className=gain>0?"positive":"negative";strong.textContent=`${Math.abs(gain).toFixed(2)}% ${gain>0?"lower":"higher"} error`;
  const label=document.createElement("span");label.textContent="vs Framewise calibration";
  byId("sim-result-gain").replaceChildren(strong,label);
  byId("sim-result-note").textContent=split==="test"?"Held-out appearance–dynamics composition (o2, d2), evaluated at the trained five-block horizon. All methods receive the same forecasting inputs.":"Mean over three extrapolation settings: a new appearance, new dynamics, and both together. The five-block forecast horizon stays fixed.";
}
function currentExample(){return showcase?.benchmarks[demoBenchmark]?.examples.find(e=>e.id===byId("sim-example").value);}
function stopPlayback(){clearInterval(playbackTimer);playbackTimer=null;byId("sim-play").setAttribute("aria-label","Play recorded rollout");byId("sim-play").querySelector("span").textContent="Play";byId("sim-play").classList.remove("playing");}
function renderStep(){
  const e=currentExample();if(!e)return;
  const call=e.timeline[Number(byId("sim-step").value)];
  byId("sim-call").textContent=`Native call ${call}`;
  for(const [mode,prefix] of [["factorized","sim-ours"],["framewise","sim-baseline"]]){
    const m=e.methods[mode];const frame=m.frames.filter(f=>f.native_call<=call).at(-1);
    const image=byId(prefix);image.src=frame.src;
    image.alt=`${mode==="factorized"?"Historical context model":"Framewise calibration"} recorded ${names[demoBenchmark]} observation at native call ${frame.native_call}`;
    const ended=call>=m.stop_call;
    byId(`${prefix}-time`).textContent=ended?`${m.success?"Goal reached":"Budget ended"} · call ${m.stop_call}`:`Observed · call ${frame.native_call}`;
    byId(`${prefix}-time`).className=`frame-label ${ended&&m.success?"positive":""}`;
  }
}
function renderExample(){
  stopPlayback();const e=currentExample();if(!e)return;
  byId("sim-goal").src=e.goal;
  byId("sim-goal").alt=`Shared ${names[demoBenchmark]} goal image for ${e.trajectory_id}`;
  byId("sim-task").textContent=`${e.trajectory_id} · development · seed 0`;
  byId("sim-step").max=e.timeline.length-1;byId("sim-step").value=0;
  byId("sim-story-title").textContent=e.category==="ours_only"?"Reaching the requested goal":e.category==="baseline_only"?"A challenge to learn from":"A difficult shared case";
  byId("sim-story").textContent= e.category==="ours_only" ? (demoBenchmark==="pusht"?"The historical context model reaches the required block and pusher configuration. Framewise continues to the action budget.":"The historical context model reaches the requested arm configuration in 23 native calls. Framewise continues to the action budget.") : e.category==="baseline_only" ? "Framewise reaches the goal in this selected case. Inspect both executions to see where further robustness is needed." : "Neither method reaches the goal within the same budget. The recorded endpoints expose a useful failure case.";
  const endpoints=byId("sim-endpoints");endpoints.replaceChildren();
  for(const [mode,label] of [["factorized","Historical context model"],["framewise","Framewise calibration"]]){
    const m=e.methods[mode],block=document.createElement("div");block.className="endpoint-summary";
    const title=document.createElement("strong");title.textContent=label;
    const outcome=document.createElement("span");outcome.className=m.success?"positive":"";outcome.textContent=`${m.success?"Goal reached":"Budget ended"} · call ${m.stop_call}`;
    const metric=document.createElement("p");metric.className="fineprint";metric.textContent=Object.entries(m.endpoint_metrics).map(([k,v])=>`${k}: ${v.value.toFixed(v.unit==="rad"?3:2)} ${v.unit}`).join(" · ");
    block.append(title,outcome,metric);endpoints.append(block);
  }
  const delta=showcase.benchmarks[demoBenchmark].comparisons.find(r=>r.split==="test"&&r.reference==="framewise");
  byId("sim-forecast-gain").textContent=`${(-delta.relative_change_percent).toFixed(2)}%`;
  renderStep();
}
function chooseDemo(value){
  stopPlayback();demoBenchmark=value;byId("droid-video").pause();
  const real=value==="droid";
  byId("sim-showcase").hidden=real;byId("droid-showcase").hidden=!real;
  byId(real?"droid-showcase":"sim-showcase").setAttribute("aria-labelledby",`demo-tab-${value}`);
  document.querySelectorAll("[data-demo]").forEach(b=>{b.setAttribute("aria-selected",String(b.dataset.demo===value));b.tabIndex=b.dataset.demo===value?0:-1;});
  if(!real&&showcase){byId("sim-example").replaceChildren();showcase.benchmarks[value].examples.forEach(e=>{const opt=document.createElement("option");opt.value=e.id;opt.textContent=e.category==="ours_only"?"Historical context model reaches the goal":e.name;byId("sim-example").append(opt);});renderExample();}
}
function chooseResults(value){
  resultsBenchmark=value;byId("simulation-results").hidden=value==="droid";byId("droid-results").hidden=value!=="droid";
  byId(value==="droid"?"droid-results":"simulation-results").setAttribute("aria-labelledby",`result-tab-${value}`);
  document.querySelectorAll("[data-results]").forEach(b=>{b.setAttribute("aria-selected",String(b.dataset.results===value));b.tabIndex=b.dataset.results===value?0:-1;});renderSimResults();
}
function registerTabs(selector,attribute,select){
  const tabs=[...document.querySelectorAll(selector)];
  tabs.forEach((b,index)=>{b.addEventListener("click",()=>select(b.dataset[attribute]));b.addEventListener("keydown",event=>{
    let next;if(event.key==="ArrowRight")next=(index+1)%tabs.length;if(event.key==="ArrowLeft")next=(index+tabs.length-1)%tabs.length;if(event.key==="Home")next=0;if(event.key==="End")next=tabs.length-1;
    if(next!==undefined){event.preventDefault();tabs[next].focus();select(tabs[next].dataset[attribute]);}
  });});
}
registerTabs("[data-demo]","demo",chooseDemo);registerTabs("[data-results]","results",chooseResults);
fetch("real-results.json").then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}).then(data=>{if(data.schema_version!==1||data.completed_runs!==12||data.completed_evaluations!==48)throw Error("Incomplete results snapshot");results=data;renderResults();}).catch(error=>{byId("results-error").hidden=false;byId("results-error").textContent=`Results could not load (${error.message}). Use the complete results download below.`;byId("population-count").textContent="Results unavailable";});
fetch("showcase.json").then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}).then(data=>{if(data.schema_version!==1)throw Error("Unsupported showcase snapshot");showcase=data;chooseDemo(demoBenchmark);renderSimResults();}).catch(error=>{byId("showcase-error").hidden=false;byId("showcase-error").textContent=`Examples could not load (${error.message}). Explore the DROID recording or download the paper.`;});
byId("result-camera").addEventListener("change",renderResults);byId("result-horizon").addEventListener("change",renderResults);byId("sim-split").addEventListener("change",renderSimResults);
byId("sim-example").addEventListener("change",renderExample);byId("sim-step").addEventListener("input",()=>{stopPlayback();renderStep();});
byId("sim-endpoint").addEventListener("click",()=>{stopPlayback();byId("sim-step").value=byId("sim-step").max;renderStep();});
byId("sim-play").addEventListener("click",()=>{
  if(playbackTimer){stopPlayback();return;}if(!currentExample())return;
  if(Number(byId("sim-step").value)===Number(byId("sim-step").max))byId("sim-step").value=0;
  byId("sim-play").setAttribute("aria-label","Pause recorded rollout");byId("sim-play").querySelector("span").textContent="Pause";byId("sim-play").classList.add("playing");renderStep();
  playbackTimer=setInterval(()=>{const slider=byId("sim-step");slider.value=Number(slider.value)+1;renderStep();if(Number(slider.value)>=Number(slider.max))stopPlayback();},850);
});
document.addEventListener("visibilitychange",()=>{if(document.hidden){stopPlayback();byId("droid-video").pause();}});
byId("video-restart").addEventListener("click",()=>{const v=byId("droid-video");v.currentTime=0;v.play().catch(()=>v.focus());});
byId("playback-speed").addEventListener("change",event=>{byId("droid-video").playbackRate=Number(event.target.value);});
byId("copy-command").addEventListener("click",async()=>{let copied=false;const value=byId("demo-command").textContent;try{await navigator.clipboard.writeText(value);copied=true;}catch(_){const field=document.createElement("textarea");field.value=value;field.style.position="fixed";field.style.left="-9999px";document.body.append(field);field.select();copied=document.execCommand("copy");field.remove();}byId("copy-command").textContent=copied?"Copied":"Select text";byId("copy-feedback").textContent=copied?"Clone command copied.":"Select and copy the command.";setTimeout(()=>{byId("copy-command").textContent="Copy";},2200);});
const navigation=[...document.querySelectorAll("nav a")];
if("IntersectionObserver" in window){const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(!visible)return;navigation.forEach(a=>{if(a.hash===`#${visible.target.id}`)a.setAttribute("aria-current","location");else a.removeAttribute("aria-current");});},{rootMargin:"-15% 0px -45% 0px",threshold:[0,.25,.5]});document.querySelectorAll("main > section[id]:not(#top)").forEach(s=>observer.observe(s));}
async function connectInference(){
  try{
    const configResponse=await fetch("demo-config.json");if(!configResponse.ok)return;
    const config=await configResponse.json();if(config.schema_version!==1||!config.enabled||!config.url)return;
    const endpoint=new URL(config.url);
    if(endpoint.protocol!=="https:"||endpoint.username||endpoint.password)throw Error("Invalid demo endpoint");
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),8000);
    let health;try{const response=await fetch(new URL("health",endpoint),{signal:controller.signal,credentials:"omit",cache:"no-store"});if(!response.ok)return;health=await response.json();}finally{clearTimeout(timeout);}
    if(health.status!=="ready")return;
    byId("inference-frame").src=endpoint.href;byId("live-demo-link").href=endpoint.href;
    byId("inference-launch").href=endpoint.href;byId("inference-launch").textContent="Open live inference";byId("inference-launch").target="_blank";byId("inference-launch").rel="noopener";
    byId("live-demo").hidden=false;byId("inference-fallback").textContent="Temporary hosted CPU preview of the historical context model. Use the downloadable inference app for a persistent local setup.";
  }catch(_){/* The local application and static result explorer remain the fallback. */}
}
connectInference();
fetch("fresh-results.json").then(r=>{if(!r.ok)throw Error("Fresh report unavailable");return r.json();}).then(data=>{
  if(data.status!=="completed"||data.schema_version!==1)return;
  const c=data.primary_comparison;
  byId("fresh-gain").textContent=`${c.relative_reduction_percent.toFixed(2)}%`;
  byId("fresh-description").textContent=`Lower five-block forecast error for the calibrated historical context model versus equally calibrated Framewise, across ${c.episode_count} new recordings from ${c.session_count} sessions and ${c.training_seed_count} seeds. Both use the same train-fitted residual-calibration procedure.`;
  byId("fresh-interval").textContent=`Primary MSE: ${data.ours_mse.toFixed(6)} vs ${data.framewise_mse.toFixed(6)}. Paired 95% CI for the difference: [${signed(c.ci95[0],6)}, ${signed(c.ci95[1],6)}]. All model and calibration choices were frozen before decoding this holdout. Ten-block differences remain inconclusive.`;
  byId("fresh-finding").hidden=false;
}).catch(()=>{/* Original, fully reported comparison remains available. */});
