from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from pathlib import Path

def reset_tts_wav(audio, output_path='./outputs/clear_audio.wav')->Path:
    '''音频降噪'''
    ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
    ans(str(audio),output_path=output_path)
    return Path(output_path)

def vacal_separate(audio, output_dir='./outputs')->list[Path]:
    '''语音分离
    Return: [Path(vacal_output), Path(instrumental_output)]
    '''
    from audio_separator.separator import Separator
    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator(output_dir=output_dir)
    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
    separator.load_model()
    # Perform the separation on specific audio files without reloading the model
    output_names = {
        "Vocals": "vocals_output",
        "Instrumental": "instrumental_output",
    }
    instrumental, vacal = separator.separate(audio, output_names)
    return [Path(output_dir) / vacal, Path(output_dir) / instrumental]


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_path', type=str)
    # parser.add_argument('--output_path', type=str, default='./outputs/clear_audio.wav')
    # args = parser.parse_args()
    # reset_tts_wav(args.input_path, args.output_path)
    vacal_separate('./test/a-song.wav')