'use strict';
const $ = id => document.getElementById(id);
const state = {index:null, visible:[], uid:null, detail:null, request:0};
const make = (tag, cls, text) => {const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const normalized = text => text.normalize('NFC').toLowerCase().replace(/[\p{P}\p{S}]/gu,' ').trim().split(/\s+/u).filter(Boolean);
const pretty = value => JSON.stringify(value,null,2);
function failure(error){$('error').textContent=String(error.message||error);$('error').hidden=false;}
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw Error(`Không tải được dữ liệu (${r.status}).`);return r.json();}
function diff(reference,hypothesis){
  const a=normalized(reference),b=normalized(hypothesis),m=a.length,n=b.length;
  const dp=Array.from({length:m+1},()=>Array(n+1).fill(0));
  for(let i=0;i<=m;i++)dp[i][0]=i;for(let j=0;j<=n;j++)dp[0][j]=j;
  for(let i=1;i<=m;i++)for(let j=1;j<=n;j++)dp[i][j]=Math.min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]!==b[j-1]));
  let i=m,j=n;const out=[];
  while(i||j){if(i&&j&&dp[i][j]===dp[i-1][j-1]+(a[i-1]!==b[j-1])){out.push({type:a[i-1]===b[j-1]?'equal':'wrong',text:b[j-1],ref:a[i-1]});i--;j--;}
    else if(i&&dp[i][j]===dp[i-1][j]+1){out.push({type:'missing',text:a[--i]});}else{out.push({type:'wrong',text:b[--j],ref:''});}}
  return out.reverse();
}
function renderOverview(){
  $('overview').replaceChildren(...state.index.models.map(model=>{
    const c=make('div','overview-item');c.append(make('div','model-title',model.title));
    const line=make('div');line.append(make('strong','',model.wer_pct.toFixed(3)+'%'),make('span','metric-label','WER'));
    c.append(line,make('p','',`${model.exact} / ${model.n} câu đúng · CER ${model.cer_pct.toFixed(3)}%`));return c;
  }));
  for(const m of state.index.models){const o=make('option','',`${m.title} · câu sai`);o.value=m.key;$('condition').append(o);}
}
function filter(){
  const q=$('search').value.trim().toLowerCase().normalize('NFC'),kind=$('kind').value,mode=$('condition').value;
  const numberQuery=q.match(/^#(\d+)$/);
  state.visible=state.index.samples.filter(r=>(!q||(numberQuery?r.conversation_no===Number(numberQuery[1]):r.search_text.normalize('NFC').includes(q)))&&(!kind||r.kind===kind)&&
    (mode==='all'||mode==='any_wrong'&&Object.values(r.errors).some(n=>n>0)||mode==='all_correct'&&Object.values(r.errors).every(n=>n===0)||r.errors[mode]>0));
  $('count').textContent=`${state.visible.length} / ${state.index.samples.length} audio`;
  $('conversation-count').textContent=`${new Set(state.visible.map(r=>r.conversation_id)).size} hội thoại · Số thứ tự giữ nguyên khi lọc`;
  $('no-results').hidden=Boolean(state.visible.length);$('sample-content').hidden=!state.visible.length;
  if(!state.visible.length){++state.request;$('player').pause();}
  const fragment=document.createDocumentFragment();
  let previousConversation=null,group=null;
  for(const r of state.visible){const b=make('button','sample'+(r.uid===state.uid?' active':''));b.dataset.uid=r.uid;
    if(r.conversation_id!==previousConversation){
      group=make('div','conversation-group');group.dataset.conversationNo=String(r.conversation_no);
      group.append(make('div','conversation-heading',`${String(r.conversation_no).padStart(3,'0')} · ${r.conversation_id}`));fragment.append(group);previousConversation=r.conversation_id;
    }
    const head=make('div','sample-head'),wrong=Object.values(r.errors).filter(n=>n>0).length;
    head.append(make('span','',`Lượt ${r.turn_no} · ${r.kind==='final'?'Yêu cầu cuối':'Lịch sử'}`),make('span','status'+(!wrong?' good':''),wrong?`${wrong}/4 sai`:'4/4 đúng'));
    b.append(head,make('div','sample-snippet',r.reference));b.title=`${r.uid} · ${r.kind==='final'?'Yêu cầu cuối':'Lịch sử'}`;
    b.addEventListener('click',()=>select(r.uid));group.append(b);}
  if(!state.visible.length)fragment.append(make('p','empty','Không có kết quả phù hợp.'));
  $('sample-list').replaceChildren(fragment);
  if(state.visible.length&&!state.visible.some(r=>r.uid===state.uid))select(state.visible[0].uid);
  updateNavigation();
}
function updateNavigation(){const at=state.visible.findIndex(r=>r.uid===state.uid);$('position').textContent=at<0?'—':`${at+1} / ${state.visible.length}`;$('prev').disabled=at<=0;$('next').disabled=at<0||at===state.visible.length-1;}
function move(offset){const at=state.visible.findIndex(r=>r.uid===state.uid),row=state.visible[at+offset];if(row)select(row.uid);}
async function select(uid){
  const request=++state.request;
  try{const d=await getJSON('/api/sample/'+encodeURIComponent(uid));if(request!==state.request)return;
    state.uid=uid;state.detail=d;history.replaceState(null,'','#'+uid);
    document.querySelectorAll('.sample').forEach(b=>b.classList.toggle('active',b.dataset.uid===uid));
    $('row-title').textContent=`Hội thoại ${String(d.conversation_no).padStart(3,'0')} · ${d.conversation_id}`;
    $('row-meta').textContent=`Hội thoại ${d.conversation_no}/${d.conversation_count} · Lượt user ${d.turn_no}/${d.turn_count} · ${d.kind==='final'?'Yêu cầu cuối':'Lịch sử'} · ${d.uid}`;
    $('duration').textContent=`${d.duration_s.toFixed(2)} giây`;$('player').pause();$('player').src='/audio/'+uid;
    $('reference').textContent=d.reference;
    $('predictions').replaceChildren(...state.index.models.map(model=>{
      const p=d.predictions[model.key],c=make('article','card prediction'),head=make('div','prediction-head');
      head.append(make('div','prediction-title',model.title),make('span','metric-pill'+(p.exact?' good':''),p.exact?'Đúng nguyên câu':`WER ${p.wer_pct.toFixed(2)}% · ${p.edits} lỗi`));
      const transcript=make('p','transcript');
      for(const op of diff(d.reference,p.text)){const el=make(op.type==='wrong'?'strong':'span',op.type==='missing'?'missing':'',op.type==='missing'?`[−${op.text}]`:op.text);if(op.type==='wrong')el.title=op.ref?'Chuẩn: '+op.ref:'Từ thêm';transcript.append(el,document.createTextNode(' '));}
      const details=make('details'),raw=make('p','raw-text',p.text);details.append(make('summary','','ASR text nguyên bản'),raw);c.append(head,transcript,details);return c;
    }));
    $('output-source').textContent=d.output_source;$('output-json').textContent=pretty(d.output);
    $('assistant-reply').hidden=!d.output.text;$('assistant-reply').textContent=d.output.text||'';
    $('output-warning').hidden=!d.gold_tools_not_listed.length;
    $('output-warning').textContent='Dữ liệu gốc có tool call không nằm trong installed_apps: '+d.gold_tools_not_listed.join(', ');
    const calls=d.output.tool_calls||[];
    $('gold-calls').replaceChildren(...(calls.length?calls.map((call,i)=>{const c=make('div','call');c.append(make('div','call-name',`${i+1}. ${call.name}`),make('pre','',pretty(call.arguments)));return c;}):[make('p','empty','Không có tool call trong output chuẩn của lượt này.')]));
    $('tool-search').value='';renderTools();renderHistory();$('context-json').textContent=pretty(d.context);updateNavigation();
  }catch(error){failure(error);}
}
function renderTools(){
  const groups=state.detail.allowed_tool_groups,q=$('tool-search').value.trim().toLowerCase();
  $('tool-count').textContent=`${groups.reduce((n,g)=>n+g.tools.length,0)} TOOLS`;
  const nodes=[];
  for(const group of groups){const filtered=group.tools.filter(t=>(t.name+' '+(t.description||'')).toLowerCase().includes(q));if(!filtered.length)continue;
    const g=make('div','tool-group');g.append(make('div','tool-group-title',group.app));
    for(const tool of filtered){const d=make('details','tool-item');d.append(make('summary','',tool.name),make('p','tool-description',tool.description||'Không có mô tả trong catalog.'));if(tool.parameters)d.append(make('pre','',pretty(tool.parameters)));g.append(d);}nodes.push(g);}
  $('tools').replaceChildren(...(nodes.length?nodes:[make('p','empty','Không có tool phù hợp.')]));
}
function renderHistory(){
  const rows=state.detail.history_before;$('history-count').textContent=`${rows.length} lượt trước`;
  $('history').replaceChildren(...(rows.length?rows.map((turn,i)=>{const c=make('div','turn '+turn.role);c.append(make('div','role',`${i+1} · ${turn.role}`),make('p','',turn.audio_text||turn.text||''));
    if(turn.audio_key){const player=make('audio');player.controls=true;player.preload='none';player.src='/audio/'+turn.audio_key;c.append(player);}
    if(turn.tool_calls?.length){const details=make('details');details.append(make('summary','',`${turn.tool_calls.length} tool call trong lịch sử`),make('pre','',pretty({tool_calls:turn.tool_calls,tool_results:turn.tool_results})));c.append(details);}return c;}):[make('p','empty','Không có lượt hội thoại trước mẫu này.')]));
}
$('search').addEventListener('input',filter);$('kind').addEventListener('change',filter);$('condition').addEventListener('change',filter);
$('tool-search').addEventListener('input',()=>{if(state.detail)renderTools();});$('prev').addEventListener('click',()=>move(-1));$('next').addEventListener('click',()=>move(1));
$('copy-output').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pretty(state.detail.output));$('copy-output').textContent='Đã copy';setTimeout(()=>$('copy-output').textContent='Copy JSON',1200);}catch(e){failure(e);}});
(async()=>{try{state.index=await getJSON('/api/index');renderOverview();const hash=location.hash.slice(1);if(state.index.samples.some(r=>r.uid===hash))state.uid=hash;filter();if(state.uid)await select(state.uid);}catch(e){failure(e);}})();
