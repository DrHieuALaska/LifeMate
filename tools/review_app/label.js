'use strict';
const $=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const pretty=x=>JSON.stringify(x,null,2);
const state={index:null,visible:[],id:null,detail:null,fullInput:null,dirty:false,revision:0,saving:null,timer:null,request:0};
if(window.innerWidth<1100)document.body.classList.add('sidebar-collapsed');
$('toggle-list').setAttribute('aria-expanded',String(!document.body.classList.contains('sidebar-collapsed')));
window.matchMedia('(max-width:1099px)').addEventListener('change',event=>{document.body.classList.toggle('sidebar-collapsed',event.matches);$('toggle-list').setAttribute('aria-expanded',String(!event.matches));});
const issueNames={asr:'ASR',name:'Tên riêng',english:'Từ tiếng Anh',number:'Số',time:'Giờ / ngày',negation:'Phủ định',tool:'Chọn tool',arguments:'Tham số',context:'Ngữ cảnh',missing:'Thiếu hành động',other:'Khác'};
for(const [key,name]of Object.entries(issueNames)){const label=el('label'),input=el('input');input.type='checkbox';input.value=key;label.append(input,document.createTextNode(name));$('issues').append(label);}
function error(e){$('error').textContent=e.message||String(e);$('error').hidden=false;}
function status(text){$('save-status').textContent=text;}
async function json(url,options){const response=await fetch(url,options);const value=await response.json();if(!response.ok)throw Error(value.error||'Không đọc/lưu được dữ liệu.');return value;}
function annotation(){
  const expected=$('expected-output').value.trim();let value=null;
  if(expected){try{value=JSON.parse(expected);}catch(e){throw Error('Output mong muốn chưa đúng JSON. Sửa JSON hoặc để trống trước khi chuyển mẫu.');}}
  return {corrected_transcript:$('corrected-transcript').value,asr_judgement:$('asr-judgement').value,reasoning_judgement:$('reasoning-judgement').value,
    expected_output:value,issues:[...$('issues').querySelectorAll('input:checked')].map(n=>n.value),comment:$('comment').value,reviewed:$('is-reviewed').checked};
}
function progress(){if(!state.index)return;$('total').textContent=state.index.n;$('reviewed').textContent=`${state.index.samples.filter(r=>r.reviewed).length} / ${state.index.n}`;}
function changed(){state.dirty=true;state.revision++;status('Chưa lưu');clearTimeout(state.timer);state.timer=setTimeout(()=>save().catch(error),900);}
async function save(){
  clearTimeout(state.timer);if(state.saving){await state.saving;if(state.dirty)return save();return;}
  if(!state.dirty||!state.id)return;
  const id=state.id,revision=state.revision,note=annotation();status('Đang lưu…');
  state.saving=(async()=>{
    const result=await json('/api/label/note/'+encodeURIComponent(id),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_fingerprint:state.index.source_fingerprint,annotation:note})});
    const row=state.index.samples.find(r=>r.id===id);Object.assign(row,{reviewed:note.reviewed,asr_judgement:note.asr_judgement,reasoning_judgement:note.reasoning_judgement});
    row.search_text=[row.id,row.asr_text,note.corrected_transcript,note.comment,JSON.stringify(state.detail.prediction)].join(' ').toLowerCase();
    if(id===state.id&&revision===state.revision){state.dirty=false;state.detail.annotation=result.annotation;status('Đã lưu trên máy');$('error').hidden=true;}else status('Có thay đổi mới…');
    progress();renderList();
  })();
  try{await state.saving;}catch(e){status('Lưu chưa thành công');throw e;}finally{state.saving=null;}
  if(state.dirty)return save();
}
function renderList(){
  const fragment=document.createDocumentFragment();
  for(const r of state.visible){const b=el('button','sample'+(r.id===state.id?' active':''));b.dataset.id=r.id;const head=el('div','sample-head');
    head.append(el('span','',`${String(r.conversation_no).padStart(3,'0')} · ${r.id}`),el('span','status'+(r.reviewed?' good':''),r.reviewed?'Đã label':'Chưa xong'));
    b.append(head,el('p','sample-snippet',r.asr_text));b.addEventListener('click',()=>select(r.id).catch(error));fragment.append(b);}
  if(!state.visible.length)fragment.append(el('p','empty','Không có kết quả phù hợp.'));$('sample-list').replaceChildren(fragment);
  $('count').textContent=`${state.visible.length} / ${state.index.n}`;navigation();
}
async function filter(){
  await save();const q=$('search').value.trim().toLowerCase(),number=q.match(/^#(\d+)$/),mode=$('filter').value;
  state.visible=state.index.samples.filter(r=>(!q||(number?r.conversation_no===Number(number[1]):r.search_text.includes(q)))&&
    (mode==='all'||mode==='pending'&&!r.reviewed||mode==='reviewed'&&r.reviewed||mode==='asr_wrong'&&r.asr_judgement==='wrong'||mode==='reasoning_wrong'&&r.reasoning_judgement==='wrong'||mode==='runtime'&&r.status!=='ok'));
  $('no-results').hidden=!!state.visible.length;$('sample-content').hidden=!state.visible.length;
  if(!state.visible.length){state.request++;$('player').pause();}
  renderList();if(state.visible.length&&!state.visible.some(r=>r.id===state.id))await select(state.visible[0].id);
}
function navigation(){const i=state.visible.findIndex(r=>r.id===state.id);$('position').textContent=i<0?'—':`${i+1} / ${state.visible.length}`;$('prev').disabled=i<=0;$('next').disabled=i<0||i>=state.visible.length-1;}
async function move(offset){const i=state.visible.findIndex(r=>r.id===state.id);const next=state.visible[i+offset];if(next)await select(next.id);}
async function select(id){
  await save();const request=++state.request;const d=await json('/api/label/sample/'+encodeURIComponent(id));if(request!==state.request)return;
  if(state.dirty)await save();if(request!==state.request)return;
  state.id=id;state.detail=d;state.dirty=false;history.replaceState(null,'','#'+id);$('error').hidden=true;
  $('row-title').textContent=`Hội thoại ${String(d.conversation_no).padStart(3,'0')} · ${id}`;
  $('row-meta').textContent=`${d.conversation_no} / ${d.conversation_count} · Kaggle: ${state.index.kernel}`;
  document.querySelectorAll('audio').forEach(a=>a.pause());
  $('duration').textContent=d.duration_s.toFixed(2)+' giây';$('player').src='/public-audio/'+id;$('asr-text').textContent=d.asr.audio_text;
  document.querySelectorAll('.panel-scroll').forEach(panel=>panel.scrollTop=0);
  const note=d.annotation;$('corrected-transcript').value=note.corrected_transcript||'';$('asr-judgement').value=note.asr_judgement||'';
  $('reasoning-judgement').value=note.reasoning_judgement||'';$('expected-output').value=note.expected_output?pretty(note.expected_output):'';
  $('comment').value=note.comment||'';$('is-reviewed').checked=!!note.reviewed;
  $('issues').querySelectorAll('input').forEach(n=>n.checked=(note.issues||[]).includes(n.value));
  status(note.updated_at?'Đã lưu trên máy':'Chưa gán nhãn');
  const m=d.metrics;$('runtime').textContent=`Trạng thái chạy: ${m.status} · ${Number(m.total_seconds).toFixed(2)} giây · ${d.prediction.tool_calls.length} tool call`;
  $('prediction-json').textContent=pretty(d.prediction);$('prediction-calls').replaceChildren(...(d.prediction.tool_calls.length?d.prediction.tool_calls.map((t,i)=>{const c=el('div','call');c.append(el('div','call-name',`${i+1}. ${t.name}`),el('pre','',pretty(t.arguments)));return c;}):[el('p','empty','Model không xuất tool call.')]));
  $('tool-search').value='';renderTools();renderHistory();$('context-json').textContent=pretty(d.context);
  state.fullInput=null;$('copy-input').disabled=true;
  for(const key of ['agent-input-json','original-input-json','full-input-json','environment-json'])$(key).textContent='Đang tải input đầy đủ…';
  renderInputSummary(d);renderList();
  json('/api/label/input/'+encodeURIComponent(id)).then(data=>{
    if(state.id!==id)return;state.fullInput=data;
    $('agent-input-json').textContent=pretty(data.agent_record_after_asr);
    $('original-input-json').textContent=pretty(data.original_manifest_record);
    $('full-input-json').textContent=pretty(data);$('environment-json').textContent=pretty(data.initial_environment_state);
    $('copy-input').disabled=false;
  }).catch(e=>{if(state.id===id){for(const key of ['agent-input-json','original-input-json','full-input-json','environment-json'])$(key).textContent='Không tải được input. Hãy tải lại trang.';error(e);}});
}
function renderInputSummary(d){
  const p=d.context.user_profile||{},fields=[['Người dùng',p.name],['Địa chỉ nhà · context.user_profile.home_address',p.home_address],['Địa chỉ công ty · context.user_profile.work_address',p.work_address],['Thời gian · context.current_time',d.context.current_time],['Múi giờ · context.timezone',d.context.timezone],['Vị trí · context.location',pretty(d.context.location)]];
  $('input-summary').replaceChildren(...fields.filter(([,value])=>value!==undefined).map(([name,value])=>{const n=el('div','input-item');n.append(el('div','input-key',name),el('div','input-value',value));return n;}));
}
function toolNode(tool){const d=el('details','tool-item');d.append(el('summary','',tool.name),el('p','tool-description',tool.description||''));if(tool.parameters)d.append(el('pre','',pretty(tool.parameters)));return d;}
function renderTools(){const q=$('tool-search').value.trim().toLowerCase(),groups=state.detail.allowed_tool_groups,nodes=[];
  $('tool-count').textContent=`${groups.reduce((n,g)=>n+g.tools.length,0)} TOOLS`;
  for(const g of groups){const tools=g.tools.filter(t=>(t.name+' '+(t.description||'')).toLowerCase().includes(q));if(!tools.length)continue;const group=el('div','tool-group');group.append(el('div','tool-group-title',g.app),...tools.map(toolNode));nodes.push(group);}
  $('tools').replaceChildren(...nodes);$('meta-tools').replaceChildren(...state.detail.meta_tools.map(toolNode));
}
function renderHistory(){
  const rows=state.detail.history,audioRows=[],textRows=[];
  $('history-count').textContent=rows.length?`Lịch sử: ${rows.length} lượt. Audio ở khung 01; text và phản hồi ở bên dưới.`:'Lịch sử: 0 lượt (history = []).';
  for(const [i,t]of rows.entries()){
    if(t.audio_key){
      const item=el('div','history-audio-item');item.id='history-audio-turn-'+i;
      item.append(el('div','audio-turn-title',`Lượt ${i+1} · ${t.role} · audio lịch sử`));
      const a=el('audio');a.controls=true;a.preload='none';a.src='/public-audio/'+t.audio_key;
      item.append(a,el('div','asr-caption','ASR MODEL'),el('p','',t.asr_text||t.audio_text||''));audioRows.push(item);
      const reference=el('button','audio-reference',`Lượt ${i+1} · ${t.role}: nghe audio ở khung 01 ↗`);
      reference.addEventListener('click',()=>{const pane=$('audio-scroll');pane.scrollTo({top:pane.scrollTop+item.getBoundingClientRect().top-pane.getBoundingClientRect().top-8,behavior:'smooth'});a.focus({preventScroll:true});});
      textRows.push(reference);
    }else{
      const n=el('div','turn '+t.role);n.append(el('div','role',`${i+1} · ${t.role}`),el('p','',t.text||t.audio_text||''));
      if(t.tool_calls?.length){const details=el('details');details.append(el('summary','',t.tool_calls.length+' tool call trong lịch sử'),el('pre','',pretty({tool_calls:t.tool_calls,tool_results:t.tool_results})));n.append(details);}textRows.push(n);
    }
  }
  $('history-audio-count').textContent=audioRows.length+' đoạn';
  $('history-audio').replaceChildren(...(audioRows.length?audioRows:[el('p','empty','Không có audio lịch sử.')]));
  $('history').replaceChildren(...(textRows.length?textRows:[el('p','empty','Không có lịch sử hội thoại.')]));
}
for(const id of ['corrected-transcript','asr-judgement','reasoning-judgement','expected-output','comment','is-reviewed'])$(id).addEventListener(id.includes('judgement')||id==='is-reviewed'?'change':'input',changed);
$('issues').addEventListener('change',changed);$('save').addEventListener('click',()=>save().catch(error));
$('search').addEventListener('input',()=>filter().catch(error));$('filter').addEventListener('change',()=>filter().catch(error));
$('prev').addEventListener('click',()=>move(-1).catch(error));$('next').addEventListener('click',()=>move(1).catch(error));
$('tool-search').addEventListener('input',()=>{if(state.detail)renderTools();});
$('done-next').addEventListener('click',async()=>{try{$('is-reviewed').checked=true;changed();await save();await move(1);}catch(e){error(e);}});
$('copy-output').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pretty(state.detail.prediction));$('copy-output').textContent='Đã copy';setTimeout(()=>$('copy-output').textContent='Copy JSON',1000);}catch(e){error(e);}});
document.querySelector('a[download]').addEventListener('click',async e=>{e.preventDefault();try{await save();location.assign('/api/label/export.jsonl');}catch(err){error(err);}});
$('copy-input').addEventListener('click',async()=>{if(!state.fullInput)return;try{await navigator.clipboard.writeText(pretty(state.fullInput));$('copy-input').textContent='Đã copy';setTimeout(()=>$('copy-input').textContent='Copy JSON',1200);}catch(e){error(e);}});
$('toggle-list').addEventListener('click',()=>{document.body.classList.toggle('sidebar-collapsed');$('toggle-list').setAttribute('aria-expanded',String(!document.body.classList.contains('sidebar-collapsed')));});
$('open-notes').addEventListener('click',()=>$('notes-dialog').showModal());
$('close-notes').addEventListener('click',()=>$('notes-dialog').close());
document.addEventListener('play',event=>{if(event.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==event.target)a.pause();});},true);
$('environment-details').addEventListener('toggle',async()=>{if(!$('environment-details').open||!state.id||$('environment-json').textContent)return;const id=state.id;$('environment-json').textContent='Đang tải…';try{const data=await json('/api/label/environment/'+id);if(state.id===id)$('environment-json').textContent=pretty(data);}catch(e){if(state.id===id)$('environment-json').textContent='Không tải được state.';error(e);}});
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
(async()=>{try{state.index=await json('/api/label/index');if(state.index.valid_available===false)document.querySelector('a[href="/valid"]').hidden=true;progress();const hash=location.hash.slice(1);if(state.index.samples.some(r=>r.id===hash))state.id=hash;await filter();if(state.id&&!state.detail)await select(state.id);}catch(e){error(e);}})();
