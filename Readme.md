📸 Live Video Captioning with Audio 🎤
Real-time video captioning using a Transformer model and speaking the captions out loud using TTS!

🚀 Features
📷 Live Webcam Feed processing

🧠 Transformer-Based Image Captioning with ViT-GPT2

🗣️ Text-to-Speech (TTS) that speaks out generated captions

🎨 On-screen Caption Display with wrapped text for readability

⚡ Efficient Frame Skipping and Resizing to improve performance

💻 CUDA / CPU device support

🛠️ Requirements
Python 3.7+

PyTorch

OpenCV

pyttsx3

Transformers (🤗 Hugging Face)

🎯 How It Works
Opens your webcam 📷

Captures frames (skipping some to save resources)

Resizes frames for faster caption generation

Generates a caption 🧠 using ViT-GPT2

Speaks the caption aloud 🗣️ using pyttsx3

Displays the caption as overlay text 📝 on the live video

Press q to quit gracefully.

Make sure your webcam is connected and accessible.

🧠 Model Used
nlpconnect/vit-gpt2-image-captioning

A Vision Transformer (ViT) encoder + GPT2 decoder model

Pretrained on a large dataset for generic image captioning

📝 Notes
Frame skipping (frame_skip) can be adjusted to balance speed and caption accuracy.

Resize factor (resize_factor) helps reduce computational load.

Ensure your microphone/speakers are working for TTS output.

CUDA will be used automatically if available 🔥.

🤖 Future Ideas
🎯 Integrate Zero-Shot Object Detection

🌐 Stream captions over a network

📱 Create a mobile or lightweight version

🔊 Add multi-language support for TTS
