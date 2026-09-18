"use strict";
const byId = id => document.getElementById(id);
let results;
const signed = (n, digits=2) => `${n>=0?"+":"−"}${Math.abs(n).toFixed(digits)}`;
function reading(target, comparison, label) {
  const gain = comparison.relative_reduction_percent;
  const [lo,hi] = comparison.ci95;
  const conclusive = hi<0 || lo>0;
  target.replaceChildren();
  const number=document.createElement("strong");
  number.className=gain>0?"positive":"negative";
  number.textContent=`${Math.abs(gain).toFixed(2)}% ${gain>0?"lower":"higher"} error`;
  const name=document.createElement("span"); name.textContent=`vs ${label}`;
  const interval=document.createElement("p");interval.className="fineprint";
  interval.textContent=`95% CI, MSE difference: [${signed(lo,6)}, ${signed(hi,6)}]. ${conclusive?"Interval excludes zero.":"Interval includes zero; inconclusive."}`;
  target.append(number,name,interval);
}
function renderResults() {
  if (!results) return;
  const camera=byId("result-camera").value, horizon=Number(byId("result-horizon").value);
  const p=results.populations.find(p=>p.camera===camera&&p.horizon===horizon);
  byId("result-rows").replaceChildren();
  p.rows.forEach(row=>{
    const tr=document.createElement("tr");
    if(row.mode==="factorized")tr.className="ours-row";
    const name=document.createElement("th");name.scope="row";name.textContent=row.name;
    const mse=document.createElement("td");mse.textContent=row.mse.toFixed(6);
    const sd=document.createElement("td");sd.textContent=row.seed_sd===null?"—":row.seed_sd.toFixed(6);
    tr.append(name,mse,sd);byId("result-rows").append(tr);
  });
  byId("population-count").textContent=`${p.episodes} held-out episodes · ${p.sessions} sessions`;
  reading(byId("framewise-gain"),p.comparisons.framewise,"Framewise");
  reading(byId("persistence-gain"),p.comparisons.persistence,"persistence");
  byId("population-note").textContent=horizon===5?"Five action blocks is the trained horizon. Camera 1 is the prespecified primary comparison.":"Ten blocks extrapolates beyond the five-block training horizon. Its eligible episode population is slightly smaller.";
}
fetch("real-results.json").then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}).then(data=>{if(data.schema_version!==1||data.completed_runs!==12||data.completed_evaluations!==48)throw Error("Incomplete results snapshot");results=data;renderResults();}).catch(error=>{byId("results-error").hidden=false;byId("results-error").textContent=`Results could not load (${error.message}). Use the complete results download below.`;byId("population-count").textContent="Results unavailable";});
byId("result-camera").addEventListener("change",renderResults);
byId("result-horizon").addEventListener("change",renderResults);
byId("video-restart").addEventListener("click",()=>{const video=byId("droid-video");video.currentTime=0;video.play().catch(()=>{video.focus();});});
byId("playback-speed").addEventListener("change",event=>{byId("droid-video").playbackRate=Number(event.target.value);});
byId("copy-command").addEventListener("click",async()=>{
  let copied=false;const value=byId("demo-command").textContent;
  try{await navigator.clipboard.writeText(value);copied=true;}catch(_){const field=document.createElement("textarea");field.value=value;field.style.position="fixed";field.style.left="-9999px";document.body.append(field);field.select();copied=document.execCommand("copy");field.remove();}
  byId("copy-command").textContent=copied?"Copied":"Select text";byId("copy-feedback").textContent=copied?"Clone command copied.":"Select and copy the command.";
  setTimeout(()=>{byId("copy-command").textContent="Copy";},2200);
});
const navigation=[...document.querySelectorAll("nav a")];
if("IntersectionObserver" in window){const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(!visible)return;navigation.forEach(a=>{if(a.hash===`#${visible.target.id}`)a.setAttribute("aria-current","location");else a.removeAttribute("aria-current");});},{rootMargin:"-15% 0px -45% 0px",threshold:[0,.25,.5]});document.querySelectorAll("main > section[id]:not(#top)").forEach(s=>observer.observe(s));}
