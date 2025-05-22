from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
result = client.predict(
		audio=handle_file('https://github.com/gradio-app/gradio/raw/main/test/a-song.wav'),
		language="auto",
		model_dir="iic/SenseVoiceSmall",
		api_name="/process"
)
print(result)