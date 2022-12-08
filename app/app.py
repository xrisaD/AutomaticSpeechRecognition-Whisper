from transformers import pipeline
import gradio as gr
import pytube as pt

pipe = pipeline(model="Hoft/whisper-small-swedish-asr")  # change to "your-username/the-name-you-picked"
sa = pipeline('sentiment-analysis', model='marma/bert-base-swedish-cased-sentiment')

def get_emoji(feeling):
    if feeling == 'POSITIVE':
        return '😊'
    else:
        return '😔'
def microphone_or_file_transcribe(audio):
    text = pipe(audio)["text"]
    sa_result = sa(text)[0]
    return text, get_emoji(sa_result['label'])
    
def youtube_transcribe(url):
    yt = pt.YouTube(url)
    
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")["text"]

    sa_result = sa(text)[0]
    return text, get_emoji(sa_result['label'])


app = gr.Blocks()

microphone_tab = gr.Interface(
    fn=microphone_or_file_transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs=[gr.Textbox(label="Text"), gr.Textbox(label="Feeling")],
    title="Whisper Small Swedish: Microphone ",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model and Sentiment Analysis.",
)

youtube_tab = gr.Interface(
    fn=youtube_transcribe, 
    inputs=[gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video", label="URL")], 
    outputs=[gr.Textbox(label="Text"), gr.Textbox(label="Feeling")],
    title="Whisper Small Swedish: Youtube",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model and Sentiment Analysis.",
)

file_tab = gr.Interface(
    fn=microphone_or_file_transcribe, 
    inputs= gr.inputs.Audio(source="upload", type="filepath"), 
    outputs=[gr.Textbox(label="Text"), gr.Textbox(label="Feeling")],
    title="Whisper Small Swedish: File",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model and Sentiment Analysis.",
)

with app:
    gr.TabbedInterface([microphone_tab, youtube_tab, file_tab], ["Microphone", "YouTube", "File"])

app.launch(enable_queue=True)
