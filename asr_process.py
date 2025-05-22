from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import pprint
from clear import reset_tts_wav, vacal_separate
from segment import segment_audio
import soundfile
from pathlib import Path

whisper_model_dir = "iic/Whisper-large-v3-turbo"
model_dir = "iic/SenseVoiceSmall"


def process_audios(audios, model_dir='iic/SenseVoiceSmall', language="auto", batch_size=1, merge_vad=False, merge_length_s=15):
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )
    rtn = []
    for audio in audios:
        res = model.generate(
            input=audio,
            cache={},
            language=language,
            use_itn=True,
            batch_size=batch_size,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
            output_timestamp=True,
        )
        rtn.append({
            'audio': audio,
            'text': rich_transcription_postprocess(res[0]["text"]),
            'timestamp': res[0]['timestamp']
        })
    return rtn

def predict_timestamps(wav_file, text_file):
    model = AutoModel(model="fa-zh")
    res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
    print(res)



def transcribe(audio, output_folder, model_dir='iic/SenseVoiceSmall', language="auto", batch_size=1, merge_vad=False, merge_length_s=15):
    fname = Path(audio).stem
    segments_dir = output_folder / 'segments'
    vacal_file, instrumental_file= vacal_separate(audio, output_folder)
    clear_audio = reset_tts_wav(vacal_file, output_path=output_folder / f'{fname}_clear_audio.wav')
    seg_audios = segment_audio(clear_audio, out=segments_dir)
    print(seg_audios)
    texts = process_audios(seg_audios, model_dir, language, batch_size, merge_vad, merge_length_s)
    words = []
    sents = []
    total_dur = 0
    seg_times = []

    for text, seg in zip(texts, seg_audios):        
        assert len(text['timestamp']) == len(text['text'])
        for w, t in zip(text['text'], text['timestamp']):
            words.append({
            'text': w,
            'start': total_dur + t[0],
            'end': total_dur + t[1],
        })
        sents.append({
            'text': text['text'],
            'start': total_dur + text['timestamp'][0][0],
            'end': total_dur + text['timestamp'][-1][1],
        })
        total_dur += int(soundfile.info(seg).duration * 1000)
        seg_times.append(total_dur)
    
    return {
        'words': words,
        'sents': sents,
        'instrumental_file': instrumental_file.as_posix(),
        'vacal_file': vacal_file.as_posix(),
        'clear_audio': clear_audio.as_posix(),
        'seg_times': seg_times,
        'srt': srt_format(sents),
    }


def format_timestamp(milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    assert milliseconds >= 0, "non-negative timestamp expected"
    print(type(milliseconds))
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    if always_include_hours or hours > 0:
        rtn = f"{hours:02d}:" 
    else:
        "" 
    rtn += f"{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    return rtn

def srt_format(sents):
    srt = ''
    for i, sent in enumerate(sents, start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(sent['start'], always_include_hours=True, decimal_marker=',')} --> "
        srt += f"{format_timestamp(sent['end'], always_include_hours=True, decimal_marker=',')}\n"
        srt += f"{sent['text']}\n\n"
    return srt

if __name__ == '__main__':
    ts = transcribe(r"D:/Download/maikase.mp3")
    # ts = transcribe(r"D:/Download/maikase.mp3", model_dir='Whisper-large-v3-turbo')
    pprint.pprint(ts)


# from funasr import AutoModel

# model = AutoModel(model="ct-punc")
# res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
# print(res)

# from funasr import AutoModel

# model = AutoModel(model="fa-zh")
# wav_file = r"D:\dev\aiclips2\aiclips\c2-The Ocean MidwaterTEMP_MPY_wvf_snd.mp3"
# text_file = './text.txt'
# res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
# print(res)