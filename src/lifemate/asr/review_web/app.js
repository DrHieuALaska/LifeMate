'use strict';
const $ = id => document.getElementById(id);
const labels = {name:'Tên riêng',number:'Số / số tiền',time:'Ngày / giờ',negation:'Phủ định / đổi ý',missing:'Thiếu từ',extra:'Thêm từ',english:'Từ tiếng Anh',reference:'Reference',noise:'Âm thanh / nhiễu',other:'Khác'};
const fields = [...document.querySelectorAll('[data-note]')];
let data = null, notes = {}, filtered = [], position = 0, view = 'all', dirty = false, timer = null, saveQueue = Promise.resolve();
for (const [key,text] of Object.entries(labels)) {
  const label = document.createElement('label'); label.className='issue';
  const input = document.createElement('input'); input.type='checkbox';input.value=key;input.dataset.issue=key;
  const span = document.createElement('span');span.textContent=text;label.append(input,span);$('issue-tags').append(label);
}
const issues = [...document.querySelectorAll('[data-issue]')];
const norm = text => text.normalize('NFC').toLocaleLowerCase('vi').replace(/[\p{P}\p{S}]/gu,' ').trim().replace(/\s+/g,' ');
const pct = value => (100*value).toLocaleString('vi-VN',{minimumFractionDigits:2,maximumFractionDigits:2})+'%';
function message(text,good=false){$('message').hidden=!text;$('message').textContent=text;$('message').className='message'+(good?' success':'');}
async function api(path,options={}){const response=await fetch(path,options);const result=await response.json();if(!response.ok)throw new Error(result.error||'Không thực hiện được');return result;}
function row(){return filtered[position];}
function noteFromFields(){const result={issues:issues.filter(x=>x.checked).map(x=>x.value)};for(const field of fields)result[field.dataset.note]=field.type==='checkbox'?field.checked:field.value;return result;}
function drawCounts(){if(!data)return;const rows=data.rows;const done=rows.filter(r=>notes[r.uid]?.listened).length;
  $('total-count').textContent=rows.length;$('count-all').textContent=rows.length;$('count-errors').textContent=rows.filter(r=>!r.predictions.hynt.exact||!r.predictions.sherpa_vi.exact).length;
  $('count-disagree').textContent=rows.filter(r=>norm(r.predictions.hynt.text)!==norm(r.predictions.sherpa_vi.text)).length;
  $('count-reviewed').textContent=done;$('count-unreviewed').textContent=rows.length-done;$('progress').max=rows.length;$('progress').value=done;$('progress-text').textContent=`Đã nghe ${done} / ${rows.length} câu`;
}
function drawList(){const container=$('audio-list');container.replaceChildren();$('list-count').textContent=`${filtered.length} audio phù hợp`;
  filtered.forEach((r,i)=>{const button=document.createElement('button');button.className='audio-item'+(i===position?' selected':'');button.dataset.uid=r.uid;
    const title=document.createElement('span');title.className='title';title.textContent=r.reference;
    const meta=document.createElement('span');meta.className='meta';const left=document.createElement('span');left.textContent=`${r.kind==='final'?'Hiện tại':'History'} · ${Number(r.duration_s).toFixed(1)}s`;
    const right=document.createElement('span');right.className=notes[r.uid]?.listened?'done':'';right.textContent=notes[r.uid]?.listened?'✓ Đã nghe':r.uid.slice(0,6);meta.append(left,right);button.append(title,meta);button.onclick=()=>navigate(i);container.append(button);});
}
function highlight(target,reference,hypothesis){const words=hypothesis.split(/\s+/),a=reference.split(/\s+/).map(norm),b=words.map(norm);const dp=Array.from({length:a.length+1},()=>new Uint16Array(b.length+1));
  for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++)dp[i][j]=a[i-1]===b[j-1]?dp[i-1][j-1]+1:Math.max(dp[i-1][j],dp[i][j-1]);
  const matched=new Set();let i=a.length,j=b.length;while(i&&j){if(a[i-1]===b[j-1]){matched.add(j-1);i--;j--;}else if(dp[i-1][j]>=dp[i][j-1])i--;else j--;}
  target.replaceChildren();words.forEach((word,k)=>{if(k)target.append(document.createTextNode(' '));const span=document.createElement('span');span.textContent=word;if(!matched.has(k))span.className='diff';target.append(span);});
}
function render(){const r=row();$('review-card').hidden=!r;$('empty').hidden=!!r;if(!r){$('player').pause();return;}
  $('row-kind').textContent=r.kind==='final'?'YÊU CẦU HIỆN TẠI':'HISTORY';$('row-position').textContent=`${position+1} / ${filtered.length}`;
  $('row-title').textContent=`Hội thoại ${r.conversation_id}`;$('row-id').textContent=`${r.uid} · ${Number(r.duration_s).toFixed(2)} giây`;
  $('reference').textContent=r.reference;$('player').pause();$('player').src='/audio/'+encodeURIComponent(r.uid);$('player').playbackRate=Number($('speed').value);$('audio-error').hidden=true;
  for(const key of ['hynt','sherpa_vi']){const pred=r.predictions[key];$(key+'-epoch').textContent=`Best theo EM · epoch ${data.models[key].epoch}`;
    $(key+'-match').textContent=pred.exact?'Khớp reference':'Có khác biệt';$(key+'-match').className='badge '+(pred.exact?'good':'warn');
    highlight($(key+'-text'),r.reference,pred.text);$(key+'-edit').textContent=`${pred.word_edits} lỗi từ / ${pred.n_words} từ reference · WER câu ${pct(pred.n_words?pred.word_edits/pred.n_words:0)}`;}
  const note=notes[r.uid]||{};for(const field of fields){if(field.type==='checkbox')field.checked=!!note[field.dataset.note];else field.value=note[field.dataset.note]||'';}
  for(const issue of issues)issue.checked=(note.issues||[]).includes(issue.value);
  $('save-status').textContent=note.updated_at?'Đã lưu trên máy · '+new Date(note.updated_at).toLocaleString('vi-VN'):'Chưa có ghi chú · tự lưu sau khi nhập';$('save-status').className='small';
  $('prev').disabled=position===0;$('next').disabled=position===filtered.length-1;drawList();
}
async function flush(){clearTimeout(timer);if(!dirty)return saveQueue;const r=row();if(!r)return;const uid=r.uid,note=structuredClone(notes[uid]);dirty=false;$('save-status').textContent='Đang lưu…';
  const task=saveQueue.catch(()=>{}).then(()=>api('/api/note',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bundle_id:data.bundle_id,uid,note})}));saveQueue=task;
  try{const saved=await task;if(JSON.stringify(notes[uid])===JSON.stringify(note))notes[uid]=saved;if(row()?.uid===uid&&!dirty){$('save-status').textContent='Đã lưu trên máy · '+new Date(saved.updated_at).toLocaleTimeString('vi-VN');$('save-status').className='small';}drawCounts();}
  catch(error){if(row()?.uid===uid)dirty=true;$('save-status').textContent='Chưa lưu được: '+error.message;$('save-status').className='small error-text';throw error;}}
function changed(){if(!row())return;notes[row().uid]=noteFromFields();dirty=true;$('save-status').textContent='Có thay đổi · đang chờ tự lưu';drawCounts();clearTimeout(timer);timer=setTimeout(()=>flush().catch(error=>message(error.message)),650);}
async function navigate(index){try{await flush();position=Math.max(0,Math.min(index,filtered.length-1));render();}catch(error){message(error.message);}}
async function filter(){try{await flush();if(!data)return;const query=norm($('search').value),kind=$('kind').value,tag=$('tag').value;
  filtered=data.rows.filter(r=>{const n=notes[r.uid]||{},p=r.predictions;return(!kind||r.kind===kind)&&(!tag||(tag==='digits'?r.contains_digits:r.english_term_hint))&&
    (!query||norm([r.uid,r.conversation_id,r.reference,p.hynt.text,p.sherpa_vi.text].join(' ')).includes(query))&&
    (view==='all'||view==='errors'&&(!p.hynt.exact||!p.sherpa_vi.exact)||view==='disagree'&&norm(p.hynt.text)!==norm(p.sherpa_vi.text)||view==='unreviewed'&&!n.listened||view==='reviewed'&&n.listened);});position=0;render();}catch(error){message(error.message);}}
async function accept(result){data=result.data;notes=result.notes||{};dirty=false;$('welcome').hidden=result.loaded;$('workspace').hidden=!result.loaded;
  $('dataset-status').textContent=result.dataset_available?'✓ Đã thấy ZIP dataset audio local.':'Chưa thấy ZIP audio local. Khởi động server với --dataset trỏ đến train+public_test.zip.';
  for(const key of ['json-export','csv-export'])$(key).disabled=!result.loaded;
  if(!data)return;$('run-description').textContent=`${data.rows.length} audio · hynt epoch ${data.models.hynt.epoch} & Sherpa epoch ${data.models.sherpa_vi.epoch}`;
  for(const [key,model]of Object.entries(data.models)){$(key+'-em').textContent=pct(model.em);$(key+'-exact').textContent=`${model.exact} / ${model.n} câu khớp sau chuẩn hóa`;$(key+'-wer').textContent=pct(model.wer);$(key+'-cer').textContent=`CER ${pct(model.cer)} · WER càng thấp càng tốt`;}
  $('run-meta').textContent=JSON.stringify({bundle_id:data.bundle_id,split_fingerprint:data.split_fingerprint,independent_test:false,models:data.models},null,2);drawCounts();await filter();}
fields.forEach(field=>field.addEventListener('input',changed));issues.forEach(field=>field.addEventListener('change',changed));
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=async()=>{try{await flush();view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b===button));await filter();}catch(error){message(error.message);}});
$('search').oninput=filter;$('kind').onchange=filter;$('tag').onchange=filter;$('prev').onclick=()=>navigate(position-1);$('next').onclick=()=>navigate(position+1);$('save').onclick=()=>flush().catch(e=>message(e.message));
$('speed').onchange=()=>{$('player').playbackRate=Number($('speed').value);};$('player').onerror=()=>{$('audio-error').textContent='Không phát được audio. Kiểm tra ZIP dataset và SHA256 bằng cách mở lại đường dẫn audio trong trình duyệt.';$('audio-error').hidden=false;};
for(const key of ['load-button','welcome-load'])$(key).onclick=()=>$('bundle-file').click();$('import-button').onclick=()=>$('notes-file').click();
$('bundle-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{await flush();message('Đang kiểm tra đủ UID, reference, metrics và ghép audio…');const result=await api('/api/load',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});await accept(result);message('Đã nạp dữ liệu. Metrics được tính lại từ predictions; ghi chú lưu riêng trên máy.',true);}catch(error){message(error.message);}finally{event.target.value='';}};
$('notes-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{await flush();notes=await api('/api/import',{method:'POST',headers:{'Content-Type':'application/json'},body:await file.text()});drawCounts();render();message('Đã nhập ghi chú đúng bundle.',true);}catch(error){message(error.message);}finally{event.target.value='';}};
for(const [id,path]of [['json-export','/api/export.json'],['csv-export','/api/export.csv']])$(id).onclick=async()=>{try{await flush();const a=document.createElement('a');a.href=path;a.download='';a.click();}catch(error){message(error.message);}};
$('details-button').onclick=()=>{$('run-details').open=!$('run-details').open;};
document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT','AUDIO'].includes(document.activeElement.tagName)||!data)return;if(event.key==='ArrowRight'){event.preventDefault();navigate(position+1);}if(event.key==='ArrowLeft'){event.preventDefault();navigate(position-1);}});
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
api('/api/data').then(accept).catch(error=>message('Không kết nối được server local: '+error.message));
