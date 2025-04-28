import cv2
import torch
import pyttsx3
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


# Initialize Text-to-Speech Engine (pyttsx3)
def initialize_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speech speed
    engine.setProperty('volume', 0.9)  # Volume level
    return engine


# Speak the generated caption
def text_to_speech(engine, text):
    engine.say(text)
    engine.runAndWait()


# Load Transformer-based captioning model
def load_captioning_model():
    gan_caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return gan_caption_model, processor, tokenizer


# Generate captions from the image frame
def generate_caption(frame, gan_caption_model, processor, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan_caption_model.to(device)

    pixel_values = processor(images=frame, return_tensors="pt").pixel_values.to(device)
    output_ids = gan_caption_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


# Utility to wrap text for OpenCV display
def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split(' ')
    lines = []
    current_line = words[0]

    for word in words[1:]:
        # Measure the width of the current line with the next word added
        line_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0][0]
        if line_size < max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


# Main function to process live video feed, generate captions, and speak them
def live_feed_captioning_with_audio(frame_skip=3, resize_factor=0.5):
    # Load the captioning model and TTS engine
    gan_caption_model, processor, tokenizer = load_captioning_model()
    tts_engine = initialize_tts_engine()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    frame_count = 0

    while True:
        # Capture frame-by-frameq
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # Skip frames to reduce processing load
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize frame to improve processing speed
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

        # Generate caption
        caption = generate_caption(frame_resized, gan_caption_model, processor, tokenizer)

        # Speak the generated caption
        print(f"Caption: {caption}")
        text_to_speech(tts_engine, caption)

        # Wrap the caption to fit within the frame width
        max_width = frame_resized.shape[1] - 20  # Leave some margin
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        wrapped_caption = wrap_text(caption, max_width, font, font_scale, thickness)

        # Display wrapped caption on the frame
        y_offset = 30
        for line in wrapped_caption:
            y_offset += 30  # Move down for each line
            cv2.putText(frame, line, (10, y_offset), font, font_scale, (255, 255, 255), thickness)

        # Show the live feed
        cv2.imshow('Live Feed Captioning with Audio', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting live feed...")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_feed_captioning_with_audio(frame_skip=3, resize_factor=0.5)
