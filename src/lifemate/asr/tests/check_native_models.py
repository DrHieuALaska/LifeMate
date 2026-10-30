"""Optional real Transformers API check with tiny RANDOM models, no downloads/data.

This verifies API compatibility only, never pretrained quality or Kaggle GPU support.
"""
import argparse
import json
from pathlib import Path
import sys
args=argparse.ArgumentParser()
args.add_argument('--extra-deps',type=Path,help='Optional isolated audio dependency directory')
args.add_argument('--check-processor',action='store_true',help='Also check the native audio processor (requires audio dependencies)')
options=args.parse_args()
if options.extra_deps: sys.path.append(str(options.extra_deps.resolve()))
import torch
import transformers
from transformers import (
    Qwen3ASRConfig,Qwen3ASRForConditionalGeneration,
    WhisperConfig,WhisperForConditionalGeneration,Wav2Vec2Config,Wav2Vec2ForCTC,
)

torch.set_num_threads(1)
torch.manual_seed(2026)
assert transformers.__version__=='5.14.1',transformers.__version__
results={}
with torch.inference_mode():
    whisper=WhisperForConditionalGeneration(WhisperConfig(
        vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=8,max_target_positions=32,bos_token_id=1,eos_token_id=2,
        pad_token_id=0,decoder_start_token_id=1,suppress_tokens=[],begin_suppress_tokens=[])).eval()
    whisper.generation_config.lang_to_id={'<|vi|>':3}
    whisper.generation_config.task_to_id={'transcribe':4}
    whisper.generation_config.no_timestamps_token_id=5
    whisper.generation_config.forced_eos_token_id=2
    generated=whisper.generate(input_features=torch.zeros(1,8,16),language='vi',task='transcribe',
        num_beams=1,do_sample=False,max_new_tokens=4,return_timestamps=False,
        return_dict_in_generate=True,output_scores=False,output_logits=False)
    assert int(generated.sequences[0,-1])==2
    results['whisper_raw_eos']=True

    qwen=Qwen3ASRForConditionalGeneration(Qwen3ASRConfig(
        audio_config=dict(num_mel_bins=8,encoder_layers=1,encoder_attention_heads=2,
            encoder_ffn_dim=32,d_model=16,output_dim=16,downsample_hidden_size=4,
            n_window=50,n_window_infer=100,max_position_embeddings=13),
        text_config=dict(model_type='qwen3',vocab_size=64,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            max_position_embeddings=64,pad_token_id=0,eos_token_id=2),
        audio_token_id=7,timestamp_token_id=8,pad_token_id=0,eos_token_id=[2,3])).eval()
    prefix=torch.tensor([[1]+[7]*13+[9]])
    generated=qwen.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),
        input_features=torch.zeros(1,8,100),input_features_mask=torch.ones(1,100,dtype=torch.long),
        num_beams=1,do_sample=False,max_new_tokens=2,forced_eos_token_id=2,
        return_dict_in_generate=True,output_scores=False,output_logits=False)
    tokens=generated.sequences[:,prefix.shape[1]:]
    assert tokens.shape[1]>0 and int(tokens[0,-1]) in [2,3]
    results['qwen_raw_generation_and_prefix_slice']=True

    ctc=Wav2Vec2ForCTC(Wav2Vec2Config(vocab_size=16,hidden_size=16,num_hidden_layers=1,
        num_attention_heads=2,intermediate_size=32,conv_dim=(8,8,8),conv_stride=(5,2,2),
        conv_kernel=(10,3,3),num_conv_pos_embedding_groups=2,pad_token_id=0,
        mask_time_prob=0.)).eval()
    ids=ctc(input_values=torch.zeros(1,1600)).logits.argmax(dim=-1)
    assert ids.ndim==2 and ids.shape[-1]>0
    results['wav2vec2_greedy_forward']=True
print(json.dumps(dict(transformers=transformers.__version__,torch=torch.__version__,
    device='cpu',random_weights_only=True,checks=results),indent=2))

# This optional check uses the REAL native processor with a locally built toy tokenizer.
# No pretrained tokenizer, weights or competition audio are downloaded.
if options.extra_deps or options.check_processor:
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3ASRProcessor,WhisperFeatureExtractor,PreTrainedTokenizerFast
    vocab={'<pad>':0,'<unk>':1,'<eos>':2,'Vietnamese':3,'<audio_bos>':4,
        '<audio_eos>':5,'reserved':6,'<audio>':7,'<asr_text>':8,'synthetic':9,'transcript':10,'language':11}
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>',eos_token='<eos>',
        additional_special_tokens=['<audio_bos>','<audio_eos>','<audio>'])
    tokenizer.audio_token='<audio>';tokenizer.audio_bos_token='<audio_bos>';tokenizer.audio_eos_token='<audio_eos>'
    template="{% for message in messages %}{% for item in message.content %}{% if item.type == 'audio' %}<audio_bos><audio><audio_eos>{% elif item.type == 'text' %}{{ item.text }} {% endif %}{% endfor %}{% endfor %}"
    processor=Qwen3ASRProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=8,sampling_rate=16000,chunk_length=1),
        tokenizer=tokenizer,chat_template=template)
    inputs=processor.apply_transcription_request(audio=np.zeros(16000,dtype='float32'),language='vi')
    assert inputs['input_features'].shape[1]==8
    assert int((inputs['input_ids']==7).sum())==13
    for name,value in inputs.items(): assert isinstance(value,torch.Tensor),(name,type(value))
    # Feed the real processor output through the real tiny random model as the adapter does.
    with torch.inference_mode():
        generated=qwen.generate(**inputs,max_new_tokens=2,num_beams=1,do_sample=False,
            forced_eos_token_id=2,return_dict_in_generate=True,output_scores=False,output_logits=False)
    suffix=generated.sequences[:,inputs['input_ids'].shape[1]:]
    assert suffix.shape[1]>0 and int(suffix[0,-1])==2
    parsed=processor.decode(torch.tensor([[11,3,8,9,10,2]]),return_format='transcription_only')
    assert parsed==['synthetic transcript'],parsed
    print('Real Qwen native processor: synthetic waveform, chat template, token lengths and transcription parsing OK')
