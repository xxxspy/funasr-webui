from asr_process import *
import gradio as gr
from pathlib import Path
import pprint

def process(audio, language="auto", model_dir='iic/SenseVoiceSmall'):
    output_folder = Path('./outputs')
    output_folder.mkdir(parents=True, exist_ok=True)
    data = transcribe(audio, output_folder, model_dir=model_dir, language=language, batch_size=10)
    pprint.pprint(data)
    return [data['words'], data['sents'], data['srt'], data['instrumental_file'], data['vacal_file'],data['clear_audio'],data['seg_times'], output_folder.absolute().as_posix(),]

# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎙️ 音频转录工具 (FunAsr webUI) by DataSense https://space.bilibili.com/383752637
        上传一个音频文件或使用麦克风录音，选择语言和模型大小，然后获取 JSON 格式的转录结果。
        **注意:** 首次选择某个模型时，可能需要一些时间来下载模型文件。 或者先下载全部模型文件: [下载](https://pan.quark.cn/s/9a8b52e4cd22)。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",  # Whisper 需要文件路径
                label="上传音频文件或录音 (Upload Audio or Record)"
            )
            model_dropdown = gr.Dropdown(
                choices=list(['iic/SenseVoiceSmall']),
                value="iic/SenseVoiceSmall", # 默认值
                label="选择模型 (Model)"
            )
            language_dropdown = gr.Dropdown(
                choices=list(['auto', 'en', 'zh']),
                value="auto",
                label="选择语言 (Language)"
            )
            submit_button = gr.Button("开始转录 (Transcribe)", variant="primary")

        with gr.Column(scale=2):
            output_fpath = gr.Text(label="输出文件路径")
            instrumental_file = gr.Audio(label="背景音乐")
            vacal_file = gr.Audio(label="高清语音")
            clear_audio = gr.Audio(label="高清语音")
            with gr.Accordion('JSON 数据', open=False):
                seg_times = gr.JSON(label="分段时间戳", )
                word_json = gr.JSON(label="单词 JSON 格式")
                sent_json = gr.JSON(label="句子 JSON 格式")
            srt_text = gr.Textbox(label="SRT 格式 (SubRip)")
        

    submit_button.click(
        fn=process,
        inputs=[audio_input, language_dropdown, model_dropdown],
        outputs=[word_json, sent_json, srt_text, instrumental_file, vacal_file, clear_audio, seg_times, output_fpath]
    )


if __name__ == "__main__":
    app.launch(debug=True, inbrowser=True) # debug=True 可以在终端看到更详细的日志