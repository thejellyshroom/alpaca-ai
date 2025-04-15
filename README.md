[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/x92FLnwf-nA)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullahalasif-bd/)
[![GitHub Stars](https://img.shields.io/github/stars/asiff00/On-Device-Speech-to-Speech-Conversational-AI?style=for-the-badge)](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/asiff00/On-Device-Speech-to-Speech-Conversational-AI?style=for-the-badge)](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI/network/members)
# On Device Speech to Speech Conversational AI
[![On Device Speech to Speech AI](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI/raw/main/assets/system_architecture.svg)](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI/raw/main/assets/video_demo.mov)

This is a real-time conversational system for two-way speech communication with AI models, utilizing a continuous streaming architecture for fluid conversations with immediate responses and natural interruption handling. All components of this system are run locally [on CPU, in my test system].

<h2 style="color: yellow;">HOW TO RUN IT</h2>

1. **Prerequisites:**
   - Install Python 3.8+ (tested with 3.12)
   - Install [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0) / `sudo apt install -y espeak-ng` for Linux (required for voice synthesis) [Linux user check this issue](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI/issues/7#issuecomment-2661541707)
   - Install Ollama from https://ollama.ai/

2. **Setup:**
   - Clone this repository `git clone https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI.git`
   - Run `cd On-Device-Speech-to-Speech-Conversational-AI` and go to the project directory
   - Run `git lfs pull` from the project root directory to download the models and voices
   - Copy `.env.template` to `.env`
   - Add your HuggingFace token to `.env`
   - Install requirements: `pip install -r requirements.txt`
   - Add any missing packages if not already installed `pip install <package_name>`


4. **Run Ollama:**
   - Start Ollama service
   - Run: `ollama run qwen2.5:0.5b-instruct-q8_0` or any other model of your choice

5. **Start Application:**
   - Run: `python speech_to_speech.py`
   - Wait for initialization (models loading)
   - Start talking when you see "Voice Chat Bot Ready"
   - Long press `Ctrl+C` to stop the application
</details>


# How does it work?
We basically put a few models together to work in a multi-threaded architecture, where each component operates independently but is integrated through a queue management system to ensure performance and responsiveness. The flow works as follows: 
### Loop (VAD -> Whisper -> LM -> TextChunker -> TTS)

To achieve that we use:

- **Voice Activity Detection**: Pyannote:pyannote/segmentation-3.0
- **Speech Recognition**: Whisper:whisper-tiny.en (OpenAI)
- **Language Model**: LM Studio/Ollama with qwen2.5:0.5b-instruct-q8_0
- **Voice Synthesis**: Kokoro:hexgrad/Kokoro-82M

We use custom text processing and queues to manage data, with separate queues for text and audio. This setup allows the system to handle heavy tasks without slowing down. We also use an interrupt mechanism allowing the user to interrupt the AI at any time. This makes the conversation feel more natural and responsive rather than just a generic TTS engine.

## Demo Video:
A demo video is uploaded here. Either click on the thumbnail or click on the YouTube link: [https://youtu.be/x92FLnwf-nA](https://youtu.be/x92FLnwf-nA).

[![On Device Speech to Speech AI Demo](https://img.youtube.com/vi/x92FLnwf-nA/0.jpg)](https://youtu.be/x92FLnwf-nA)

## Performance:
![Timing Chart](assets/timing_chart.png)

I ran this test on an AMD Ryzen 5600G, 16 GB, SSD, and No-GPU setup, achieving consistent ~2s latency. On average, it takes around 1.5s for the system to respond to a user query from the point the user says the last word. Although I haven't tested this on a GPU, I believe testing on a GPU would significantly improve performance and responsiveness.

## How do we reduce latency?
### Priority based text chunking 
We capitalize on the streaming output of the language model to reduce latency. Instead of waiting for the entire response to be generated, we process and deliver each chunk of text as soon as they become available, form phrases, and send it to the TTS engine queue. We play the audio as soon as it becomes available. This way, the user gets a very fast response, while the rest of the response is being generated.

Our custom `TextChunker` analyzes incoming text streams from the language model and splits them into chunks suitable for the voice synthesizer. It uses a combination of sentence breaks (like periods, question marks, and exclamation points) and semantic breaks (like "and", "but", and "however") to determine the best places to split the text, ensuring natural-sounding speech output.

The `TextChunker` maintains a set of break points:
- **Sentence breaks**: `.`, `!`, `?` (highest priority)
- **Semantic breaks** with priority levels:
  - Level 4: `however`, `therefore`, `furthermore`, `moreover`, `nevertheless`
  - Level 3: `while`, `although`, `unless`, `since`
  - Level 2: `and`, `but`, `because`, `then`
- **Punctuation breaks**: `;` (4), `:` (4), `,` (3), `-` (2)

When processing text, the `TextChunker` uses a priority-based system:
1. Looks for sentence-ending punctuation first (highest priority 5)
2. Checks for semantic break words with their associated priority levels
3. Falls back to punctuation marks with lower priorities
4. Splits at target word count if no natural breaks are found

The text chunking method significantly reduces perceived latency by processing and delivering the first chunk of text as soon as it becomes available. Let's consider a hypothetical system where the language model generates responses at a certain rate. If we imagine a scenario where the model produces a response of N words at a rate of R words per second, waiting for the complete response would introduce a delay of N/R seconds before any audio is produced. With text chunking, the system can start processing the first M words as soon as they are ready (after M/R seconds), while the remaining words continue to be generated. This means the user hears the initial part of the response in just M/R seconds, while the rest streams in naturally.

### Leading filler word LLM Prompting
We use a another little trick in the LLM prompt to speed up the system’s first response. We ask the LLM to start its reply with filler words like “umm,” “so,” or “well.” These words have a special role in language: they create natural pauses and breaks. Since these are single-word responses, they take only milliseconds to convert to audio. When we apply our chunking rules, the system splits the response at the filler word (e.g., “umm,”) and sends that tiny chunk to the TTS engine. This lets the bot play the audio for “umm” almost instantly, reducing perceived latency. The filler words act as natural “bridges” to mask processing delays. Even a short “umm” gives the illusion of a fluid conversation, while the system works on generating the rest of the response in the background. Longer chunks after the filler word might take more time to process, but the initial pause feels intentional and human-like.

We have fallback plans for cases when the LLM fails to start its response with fillers. In those cases, we put hand breaks at 2 to 5 words, which comes with a cost of a bit of choppiness at the beginning but that feels less painful than the system taking a long time to give the first response.

**In practice,** this approach can reduce perceived latency by up to 50-70%, depending on the length of the response and the speed of the language model. For example, in a typical conversation where responses average 15-20 words, our techniques can bring the initial response time down from 1.5-2 seconds to just `0.5-0.7` seconds, making the interaction feel much more natural and immediate.


## Resources
This project utilizes the following resources:
*   **Text-to-Speech Model:** [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
*   **Speech-to-Text Model:** [Whisper](https://huggingface.co/openai/whisper-tiny.en)
*   **Voice Activity Detection Model:** [Pyannote](https://huggingface.co/pyannote/segmentation-3.0)
*   **Large Language Model Server:** [Ollama](https://ollama.ai/)
*   **Fallback Text-to-Speech Engine:** [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

## Acknowledgements
This project draws inspiration and guidance from the following articles and repositories, among others:
*   [Realtime speech to speech conversation with MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)
*   [A Comparative Guide to OpenAI and Ollama APIs](https://medium.com/@zakkyang/a-comparative-guide-to-openai-and-ollama-apis-with-cheathsheet-5aae6e515953)
*   [Building Production-Ready TTS with Kokoro-82M](https://medium.com/@simeon.emanuilov/kokoro-82m-building-production-ready-tts-with-82m-parameters-unfoldai-98e36ff286b9)
*   [Kokoro-82M: The Best TTS Model in Just 82 Million Parameters](https://medium.com/data-science-in-your-pocket/kokoro-82m-the-best-tts-model-in-just-82-million-parameters-512b4ba4f94c)
*   [StyleTTS2 Model Implementation](https://github.com/yl4579/StyleTTS2/blob/main/models.py)
