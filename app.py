import json
import os
import streamlit as st
from streamlit import session_state
import streamlit as st
import google.generativeai as genai
from st_audiorec import st_audiorec
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from time import sleep
import smtplib
import random
import string
import re
import pypdfium2 as pdfium
import datetime
import os
import json
from PIL import Image
import os
import numpy as np
import streamlit as st
from streamlit import session_state
import streamlit as st
from pytube import YouTube
import os
import hashlib
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore 
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # type: ignore
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tensorflow.keras.models import load_model  # type: ignore
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from pytube import YouTube
from deep_translator import GoogleTranslator
from gtts import gTTS
from string import punctuation
from heapq import nlargest
import requests
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
from googlesearch import search
import subprocess
import time
from caption import generate_caption
import io
import datetime

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

load_dotenv()


st.set_page_config(
    page_title="Smart Visual Learning Portal",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
     }
)


MODEL_DIRECTORY = "imagecaption//blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)
VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.mp3"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")
SENDER_MAIL_ID = os.getenv("SENDER_MAIL_ID")

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


@st.cache_resource
def load_lstm_models():
    encoder_model = load_model(
        "AbstractiveTextSummarization//LSTM//encoder_model.keras"
    )
    decoder_model = load_model(
        "AbstractiveTextSummarization//LSTM//decoder_model.keras"
    )
    # Load tokenizer from file
    with open("AbstractiveTextSummarization//LSTM//x_tokenizer.json", "r") as json_file:
        tokenizer_json = json_file.read()
        x_tokenizer = tokenizer_from_json(tokenizer_json)
    # Load tokenizer from file
    with open("AbstractiveTextSummarization//LSTM//y_tokenizer.json", "r") as json_file:
        tokenizer_json = json_file.read()
        y_tokenizer = tokenizer_from_json(tokenizer_json)
    return encoder_model, decoder_model, x_tokenizer, y_tokenizer

@st.cache_resource
def load_t5_models():
    model = T5ForConditionalGeneration.from_pretrained(
        "AbstractiveTextSummarization//T5//t5_model.pt"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        "AbstractiveTextSummarization//T5//t5_tokenizer.pt"
    )
    return model, tokenizer


def smart_tutor(question, previous_lectures, previous_questions):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        prompt = f"""
        As a smart tutor, your task is to provide detailed answers to the student's questions based on the provided lectures and questions. You should be able to answer follow-up questions and provide additional information as needed. You should also be able to answer any questions that the student may have about the lectures or questions provided. The student may ask questions about the topics discussed in the lectures or the questions provided. You should provide detailed responses to the student's questions without revealing that you are an AI assistant. You should also be able to conduct internet searches to gather relevant results if needed.
        
        Previous Questions: {previous_questions}
        Previous Lectures: {previous_lectures}
        
        """

        if len(prompt) > 16385:
            prompt = prompt[:16385]
        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-0125",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def smart_tutor_gemini(question, previous_lectures, previous_questions):
    try:

        genai.configure(api_key=GEMINI_API_KEY)
        # The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"""
        As a smart tutor, your task is to provide detailed answers to the student's questions based on the provided lectures and questions. You should be able to answer follow-up questions and provide additional information as needed. You should also be able to answer any questions that the student may have about the lectures or questions provided. The student may ask questions about the topics discussed in the lectures or the questions provided. You should provide detailed responses to the student's questions without revealing that you are an AI assistant. You should also be able to conduct internet searches to gather relevant results if needed.
        
        Previous Questions: {previous_questions}
        Previous Lectures: {previous_lectures}
        
        """
        if len(prompt) > 16385:
            prompt = prompt[:16385]

        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def generate_search_terms(Highlights):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        prompt = (
            """
As a smart teaching assistant, your task is to generate relevant search terms based on the provided highlights for the previous teaching sessions. You should identify the key topics discussed during the session and generate search terms that can be used to find more information on those topics. Do not mention that you are an AI assistant.

Your responsibilities also include:
- Identifying the main topics discussed in the class.
- Generating search terms based on those topics.
- Search term should be in the form of a list separated by newline python characters ordered from most to least relevant topics.
- Format: 
search term 1
search term 2
search term 3
..
- Do not add anythig else in the response."""
            + Highlights
        )
        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-0125",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def smart_highlights(session_transcription):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        prompt = (
            """
As a smart teaching assistant, your task is to generate detailed class session highlights based on the provided session transcription. Additionally, you are responsible for conducting relevant internet research to supplement the teaching with necessary information. This includes details about discussed topics such as Homework, Assignments, Projects,Food for thoughts, Higher Order thinking questions, and other important information. You should provide a detailed summary of the session and highlight key takeaways without revealing that you are an AI assistant.
Your responsibilities also include:
- Identifying the main topics discussed in the class.
- Executing possible insights from the provided data.
- Providing additional information or answering questions from the user.
- Conducting internet searches to gather relevant results if the meeting involves gathering information on a specific topic.

Session Transcription:
"""
            + session_transcription
        )

        prompt += session_transcription
        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            # model="gpt-3.5-turbo",
            model="gpt-3.5-turbo-0125",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def summarize_lstm(input_text):
    encoder_model, decoder_model, x_tokenizer, y_tokenizer = load_lstm_models()
    x = np.array([input_text])
    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index
    max_len_text = 80
    max_len_summary = 10

    def decode_sequence(input_seq):
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_word_index["sostok"]
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + [e_out, e_h, e_c]
            )
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            if sampled_token != "eostok":
                decoded_sentence += " " + sampled_token
            if sampled_token == "eostok" or len(decoded_sentence.split()) >= (
                max_len_summary - 1
            ):
                stop_condition = True
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            e_h, e_c = h, c
        return decoded_sentence

    def seq2summary(input_seq):
        newString = ""
        for i in input_seq:
            if (i != 0 and i != target_word_index["sostok"]) and i != target_word_index[
                "eostok"
            ]:
                newString = newString + reverse_target_word_index[i] + " "
        return newString

    def seq2text(input_seq):
        newString = ""
        for i in input_seq:
            if i != 0:
                newString = newString + reverse_source_word_index[i] + " "
        return newString

    x = pad_sequences(
        x_tokenizer.texts_to_sequences(x), maxlen=max_len_text, padding="post"
    )
    return decode_sequence(x.reshape(1, max_len_text))


def summarize_t5(text):
    model, tokenizer = load_t5_models()
    max_length = len(text) // 2
    inputs = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = model.generate(
        inputs.to(model.device),
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarize_genAi(text):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        prompt = f"""
As a smart assistant, your task is to summarize the provided text. You should generate a concise and coherent summary of the text without losing its key points. The summary should be clear, concise, and relevant to the original text. You should also ensure that the summary is well-structured and easy to understand. The summary should be in your own words and should not include any verbatim text from the original text. You should also avoid adding any new information that is not present in the original text.
"""
        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-0125",
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def summarize_spacy(text_content):
    nlp = en_core_web_sm.load()
    stop_words = list(STOP_WORDS)
    punctuation_items = punctuation + "\n"
    nlp = spacy.load("en_core_web_sm")

    nlp_object = nlp(text_content)
    word_frequencies = {}
    for word in nlp_object:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation_items:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_token = [sentence for sentence in nlp_object.sents]
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.text.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    # Getting the 10% of the sentences with the maximum scores
    select_length = int(len(sentence_token) * 0.4)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = " ".join(final_summary)
    return summary


def stream_data(text):
    for line in text.split("\n"):
        for word in line.split(" "):
            yield word + " "
            time.sleep(0.02)
        yield "\n"


def get_summary(text_content, model):
    if model == "Spacy Summarization":
        return summarize_spacy(text_content)
    elif model == "T5 Summarization":
        return summarize_t5(text_content)
    elif model == "LSTM Summarization":
        return summarize_lstm(text_content)
    elif model == "GenAI Summarization":
        return summarize_genAi(text_content)
    return None


def get_details_from_image(image, question):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            image = Image.open(image)
        except:
            pass
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"""
            Your task is to analyze the provided image and provide the corresponding details. You should be able to identify the objects, people, and activities in the image and provide a detailed description.
            You should also be able to analyse any graphical elements, such as charts, graphs, or diagrams, and provide relevant information.
            You should also be able to provide additional information about the image, such as the location, time, and context of the image. You should provide a detailed and accurate analysis of the image without revealing that you are an AI assistant. You should also be able to answer any follow-up questions that the user may have about the image.
            
            
            Question: {question}
        """
        response = model.generate_content(
            [
                prompt,
                image,
            ],
            safety_settings=safety_settings,
        )
        return response.text
    except Exception as e:
        st.error(f"Error getting details from image: {e}")
        return None


def get_transcript_from_video(VIDEO_PATH):
    try:
        command = f"ffmpeg -i {VIDEO_PATH} -vn -acodec mp3 {AUDIO_PATH}"
        os.system(command)
        transcription = get_transcription_from_audio(AUDIO_PATH)
        # if os.path.exists(AUDIO_PATH):
        #     os.remove(AUDIO_PATH)
        return transcription
    except Exception as e:
        st.error("Error in fetching transcript {e}")
        return None


def get_transcript_from_url(url):
    try:
        url_data = urlparse(url)
        id = url_data.query[2::]
        script = YouTubeTranscriptApi.get_transcript(id)
        transcript = ""
        for text in script:
            t = text["text"]
            if t != "[Music]":
                transcript += t + " "
        return transcript
    except:
        try:
            try:
                yt = YouTube(url)
            except:
                return "Connection Error"
            stream = yt.streams.get_by_itag(251)
            stream.download("", AUDIO_PATH)
            result = get_transcription_from_audio(AUDIO_PATH)
            return result

        except Exception as e:
            st.error(f"Error in fetching transcript from URL: {e}")
            return None


def get_transcription_from_audio(file_path):
    try:
        client = OpenAI()
        audio_file = open(file_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        return transcription.text
    except Exception as e:
        st.error(f"Error in fetching transcription {e}")
        return None


def get_text_from_pdf(pdf_file):
    try:
        pdf = pdfium.PdfDocument(pdf_file)
        text = ""
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text += textpage.get_text_range()
        return text
    except Exception as e:
        print(f"Error getting text from pdf: {e}")
        return None


def get_details_from_pdf(pdf_path):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        pdf_text = get_text_from_pdf(pdf_path)
        if not pdf_text:
            return "Error extracting text from PDF."
        print(f"Uploading file...")
        file = genai.upload_file(path=pdf_path)
        print(f"Completed upload: {file.uri}")

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Initialize the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Define the prompt
        prompt = """
        You are a very professional document summarization specialist.
        Please summarize the given document in detail. Make sure to include all the key points and important information from the document.
        """
        # Combine the extracted text and the prompt
        content = f"{prompt}\n\nDocument Text:\n{pdf_text}"
        # Generate the summary
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f"Error getting details from pdf: {e}")
        return None


def translate(text, language):
    try:
        summaries = []
        while len(text) > 5000:
            index = 5000
            while text[index] != "." and index > 0:
                index -= 1
            if index == 0:
                index = 5000
            summaries.append(text[:index])
            text = text[index:]
        summaries.append(text)
        translated_summary = ""
        for s in summaries:
            translated_summary += (
                GoogleTranslator(source="auto", target=language).translate(s) + " "
            )
    except Exception:
        st.warning(f"Translation to the selected language is not supported.")
        translated_summary = text
    return translated_summary


def user_exists(email, json_file_path):
    # Function to check if user with the given email exists
    with open(json_file_path, "r") as file:
        users = json.load(file)
        for user in users["users"]:
            if user["email"] == email:
                return True
    return False


def send_verification_code(email, code):
    RECEIVER = email
    server = smtplib.SMTP_SSL("smtp.googlemail.com", 465)
    server.login(SENDER_MAIL_ID, APP_PASSWORD)
    message = f"Subject: Your Verification Code\n\nYour verification code is: {code}"
    server.sendmail(SENDER_MAIL_ID, RECEIVER, message)
    server.quit()
    st.success("Email sent successfully!")
    return True


def generate_verification_code(length=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        if (
            session_state.get("verification_code") is None
            or session_state.get("verification_time") is None
            or datetime.datetime.now() - session_state.get("verification_time")
            > datetime.timedelta(minutes=5)
        ):
            verification_code = generate_verification_code()
            session_state["verification_code"] = verification_code
            session_state["verification_time"] = datetime.datetime.now()
        if st.form_submit_button("Signup"):
            if not name:
                st.error("Name field cannot be empty.")
            elif not email:
                st.error("Email field cannot be empty.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+$", email):
                st.error("Invalid email format. Please enter a valid email address.")
            elif user_exists(email, json_file_path):
                st.error(
                    "User with this email already exists. Please choose a different email."
                )
            elif not age:
                st.error("Age field cannot be empty.")
            elif not password or len(password) < 6:  # Minimum password length of 6
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                verification_code = session_state["verification_code"]
                send_verification_code(email, verification_code)
                entered_code = st.text_input(
                    "Enter the verification code sent to your email:"
                )
                if entered_code == verification_code:
                    user = create_account(
                        name, email, age, sex, password, json_file_path
                    )
                    session_state["logged_in"] = True
                    session_state["user_info"] = user
                    st.success("Signup successful. You are now logged in!")
                elif len(entered_code) == 6 and entered_code != verification_code:
                    st.error("Incorrect verification code. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")


def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        email = email.lower()
        password = hashlib.md5(password.encode()).hexdigest()
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "exams": None,
            "highlights": None,
        }

        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")
    username = username.lower()
    password = hashlib.md5(password.encode()).hexdigest()

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information")
        st.info(f"Name: {user_info['name']}")
        st.info(f"Sex: {user_info['sex']}")
        st.info(f"Age: {user_info['age']}")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")


def main(json_file_path="data.json"):
    page = st.sidebar.selectbox(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Student Learning Hub",
            "View Previous Lectures",
            "Smart Tutor",
        ),
        key="page",
    )
    languages = {
                "English": "en",
                "Afrikaans": "af",
                "Albanian": "sq",
                "Amharic": "am",
                "Arabic": "ar",
                "Armenian": "hy",
                "Azerbaijani": "az",
                "Basque": "eu",
                "Belarusian": "be",
                "Bengali": "bn",
                "Bosnian": "bs",
                "Bulgarian": "bg",
                "Catalan": "ca",
                "Cebuano": "ceb",
                "Chichewa": "ny",
                "Chinese (simplified)": "zh-cn",
                "Chinese (traditional)": "zh-tw",
                "Corsican": "co",
                "Croatian": "hr",
                "Czech": "cs",
                "Danish": "da",
                "Dutch": "nl",
                "Esperanto": "eo",
                "Estonian": "et",
                "Filipino": "tl",
                "Finnish": "fi",
                "French": "fr",
                "Frisian": "fy",
                "Galician": "gl",
                "Georgian": "ka",
                "German": "de",
                "Greek": "el",
                "Gujarati": "gu",
                "Haitian creole": "ht",
                "Hausa": "ha",
                "Hawaiian": "haw",
                "Hebrew": "he",
                "Hindi": "hi",
                "Hmong": "hmn",
                "Hungarian": "hu",
                "Icelandic": "is",
                "Igbo": "ig",
                "Indonesian": "id",
                "Irish": "ga",
                "Italian": "it",
                "Japanese": "ja",
                "Javanese": "jw",
                "Kannada": "kn",
                "Kazakh": "kk",
                "Khmer": "km",
                "Korean": "ko",
                "Kurdish (kurmanji)": "ku",
                "Kyrgyz": "ky",
                "Lao": "lo",
                "Latin": "la",
                "Latvian": "lv",
                "Lithuanian": "lt",
                "Luxembourgish": "lb",
                "Macedonian": "mk",
                "Malagasy": "mg",
                "Malay": "ms",
                "Malayalam": "ml",
                "Maltese": "mt",
                "Maori": "mi",
                "Marathi": "mr",
                "Mongolian": "mn",
                "Myanmar (burmese)": "my",
                "Nepali": "ne",
                "Norwegian": "no",
                "Odia": "or",
                "Pashto": "ps",
                "Persian": "fa",
                "Polish": "pl",
                "Portuguese": "pt",
                "Punjabi": "pa",
                "Romanian": "ro",
                "Russian": "ru",
                "Samoan": "sm",
                "Scots gaelic": "gd",
                "Serbian": "sr",
                "Sesotho": "st",
                "Shona": "sn",
                "Sindhi": "sd",
                "Sinhala": "si",
                "Slovak": "sk",
                "Slovenian": "sl",
                "Somali": "so",
                "Spanish": "es",
                "Sundanese": "su",
                "Swahili": "sw",
                "Swedish": "sv",
                "Tajik": "tg",
                "Tamil": "ta",
                "Telugu": "te",
                "Thai": "th",
                "Turkish": "tr",
                "Ukrainian": "uk",
                "Urdu": "ur",
                "Uyghur": "ug",
                "Uzbek": "uz",
                "Vietnamese": "vi",
                "Welsh": "cy",
                "Xhosa": "xh",
                "Yiddish": "yi",
                "Yoruba": "yo",
                "Zulu": "zu",
            }

    if st.sidebar.button("Logout"):
        session_state["logged_in"] = False
        session_state["user_info"] = None
        st.success("You have been logged out successfully!")
        st.rerun()

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Student Learning Hub":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            st.title("Student Learning Hub")
            st.write("Welcome to the Student Learning Hub.")
            st.write(
                "Here you can upload various types of media to enhance your learning experience."
            )
            media_format = st.selectbox(
                "Select a type of media to upload:",
                (
                    "Image",
                    "Video",
                    "YouTube URL",
                    "PDF Document",
                    "Audio Recording",
                    "Text",
                ),
            )
            summarization_method = st.selectbox(
                "Choose summarization method",
                (
                    "Spacy Summarization",
                    "T5 Summarization",
                    "LSTM Summarization",
                    "GenAI Summarization",
                ),
            )
            transcription = None
            highlights = None
            result = None
            transcription_translated = None
            summary_translated = None
            highlights_translated = None
            
            language = st.selectbox("Select a Language", list(languages.keys()))

            if media_format == "Video":
                video = st.file_uploader(
                    "Upload Video", type=["mp4", "avi", "mov", "wmv"]
                )
                if video is not None:
                    st.video(video)
                    # save the video file
                    with open(VIDEO_PATH[:-4] + "_temp" + ".mp4", "wb") as f:
                        f.write(video.read())

                    with st.spinner("Processing video..."):
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                VIDEO_PATH[:-4] + "_temp" + ".mp4",
                                VIDEO_PATH,
                                "-y",
                            ],
                            check=True,
                        )
                    with st.spinner("Transcribing video..."):
                        transcription = get_transcript_from_video(VIDEO_PATH)
                    #if os.path.exists(VIDEO_PATH):
                     #   os.remove(VIDEO_PATH)
                    #if os.path.exists(VIDEO_PATH + "_temp"):
                    #    os.remove(VIDEO_PATH + "_temp")

            elif media_format == "YouTube URL":
                youtube_url = st.text_input("Enter YouTube URL:")
                if youtube_url:
                    st.markdown(
                        '<h1 style="color:#FF0000;">YouTube Video Summariser</h1>',
                        unsafe_allow_html=True,
                    )
                    # Fetch video title from YouTube link
                    r = requests.get(youtube_url)
                    soup = BeautifulSoup(r.text, "html.parser")
                    title = soup.title.string if soup.title else "No title available"

                    # Display video title
                    st.info(f"Video Title: {title}")

                    st.video(youtube_url)
                    with st.spinner("Transcribing YouTube video..."):
                        transcription = get_transcript_from_url(youtube_url)

            elif media_format == "PDF Document":
                pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"])
                if pdf_file is not None:
                    with open("temp.pdf", "wb") as f:
                        f.write(pdf_file.read())
                    with st.spinner("Extracting information from PDF..."):
                        transcription = get_details_from_pdf("temp.pdf")
                        # if os.path.exists("temp.pdf"):
                        #     os.remove("temp.pdf")

            elif media_format == "Audio Recording":
                record_or_upload = st.radio(
                    "Select an option", ("Record Audio", "Upload Audio")
                )
                if record_or_upload == "Record Audio":
                    st.write("Click the button below to start recording:")
                    audio = st_audiorec()
                    if audio is not None:
                        with open("temp.mp3", "wb") as f:
                            f.write(audio)
                        os.system(
                            f"ffmpeg -i temp.mp3 -codec:a libmp3lame -b:a 320k {AUDIO_PATH} -y"
                        )
                        transcription = get_transcription_from_audio(AUDIO_PATH)
                else:
                    audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])
                    if audio is not None:
                        temp_audio_path = "temp.mp3" if audio.type == "audio/mpeg" else "temp.wav"
                        with open(temp_audio_path, "wb") as f:
                            f.write(audio.read())
                        os.system(
                            f"ffmpeg -i {temp_audio_path} -codec:a libmp3lame -b:a 320k {AUDIO_PATH} -y"
                        )
                        transcription = get_transcription_from_audio(AUDIO_PATH)

            elif media_format == "Image":
                image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
                question = st.text_area("Ask a question about the image:")
                if image is not None:
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(Image.open(image))
                        st.image(image, caption=caption, use_column_width=True)
                if image is not None and question is not None and len(question) > 0:
                    with st.spinner("Extracting information from image..."):
                        transcription = get_details_from_image(image, question)

            elif media_format == "Text":
                transcription = ""
                transcription = st.text_area("Enter Text:")

            if transcription is not None:

                st.markdown(f"### Learning: \n")
                if language != "English":
                    with st.spinner("Translating transcription..."):
                        transcription_translated = translate(
                            transcription, languages[language]
                        )
                else:
                    transcription_translated = transcription
                st.write_stream(stream_data(transcription_translated))
                with st.spinner("Summarizing transcription..."):
                    summary = get_summary(transcription, summarization_method)
                    if summary:
                        result = summary
                        if language != "English":
                            translated_summary = translate(summary, languages[language])
                            if translated_summary:
                                result = translated_summary
                    else:
                        st.error("Error getting summary.")
                st.subheader("Summary:")
                if result:
                    st.write_stream(stream_data(result))
                with st.spinner("Generating highlights..."):
                    highlights = smart_highlights(transcription)
                if highlights:
                    st.subheader("Session Highlights and Takeaways:")
                    if language != "English":
                        highlights_translated = translate(
                            highlights, languages[language]
                        )
                    else:
                        highlights_translated = highlights
                    st.write_stream(stream_data(highlights_translated))
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    for user in data["users"]:
                        if user["email"] == user_info["email"]:
                            if user["highlights"] is None:
                                user["highlights"] = [
                                    {
                                        "Timestamp": datetime.datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        ),
                                        "Transcription": transcription,
                                        "Summary": summary,
                                        "Highlight": highlights,
                                        "Language": language,
                                    }
                                ]
                            else:
                                user["highlights"].append(
                                    {
                                        "Timestamp": datetime.datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        ),
                                        "Transcription": transcription,
                                        "Summary": summary,
                                        "Highlight": highlights,
                                        "Language": language,
                                    }
                                )
                            json_file.seek(0)
                            json.dump(data, json_file, indent=4)
                            json_file.truncate()

            if highlights and transcription:
                with st.spinner("Generating search terms..."):
                    search_terms = generate_search_terms(highlights).split("\n")
                    search_terms = search_terms[: (min(3, len(search_terms)))]
                st.subheader("Suggested links for further reading:")
                for search_term in search_terms:
                    searches = search(search_term, num_results=2)
                    for search_ in searches:
                        st.write(search_)
                with st.spinner("Generating audio highlights..."):
                    speech = gTTS(text=highlights, lang="en", slow=False)
                    speech.save("user_trans.mp3")
                    st.success("###  \U0001F3A7 Hear the Highlights")
                audio_file = open("user_trans.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg", start_time=0)
                audio_file.close()
                if os.path.exists("user_trans.mp3"):
                    os.remove("user_trans.mp3")

                if transcription is None:
                    transcription = ""
                if result is None:
                    result = ""
                if highlights is None:
                    highlights = ""
                data = f"""
                    Transcription:\n {transcription_translated}\n\n
                    Summary:\n {result}\n\n
                    Highlights and key takeaways:\n {highlights_translated}
                
                """

                download_button_str = f"Download Highlights and Takeaways"
                with io.BytesIO(data.encode()) as stream:
                    st.download_button(
                        download_button_str, stream, file_name="highlights.txt"
                    )
                if os.path.exists(AUDIO_PATH):
                    os.remove(AUDIO_PATH)
                if os.path.exists(VIDEO_PATH):
                    os.remove(VIDEO_PATH)

        else:
            st.warning("Please login/signup to access this page.")

    elif page == "View Previous Lectures":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            st.title("View Previous Lectures")
            st.write("Welcome to the View Previous Lectures page.")
            st.write("Here you can access previous lectures.")
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                for user in data["users"]:
                    if user["email"] == user_info["email"]:
                        user_info = user
                        break
            if user_info["highlights"] is not None and len(user_info["highlights"]) > 0:
                st.header("Previous highlights:")
                for highlight in reversed(user_info["highlights"]):
                    if highlight["Language"] != "English":
                        st.markdown(f"## {highlight['Timestamp']}")
                        if highlight["Transcription"]:
                            transcription = translate(
                            highlight["Transcription"], languages[highlight["Language"]]
                        )
                            st.markdown(f"### Transcription:")
                            st.markdown(f"{transcription}")
                        if highlight["Summary"]:
                            summary = translate(
                            highlight["Summary"], languages[highlight["Language"]]
                        )
                            st.markdown(f"### Summary:")
                            st.markdown(f"{summary}")
                        if highlight["Highlight"]:
                            highlights = translate(
                            highlight["Highlight"], languages[highlight["Language"]]
                        )
                            st.markdown(f"### Highlights:")
                            st.markdown(f"{highlights}")
                        st.write("---")
                        st.write("---")
                    else:
                        st.markdown(f"## {highlight['Timestamp']}")
                        if highlight["Transcription"]:
                            st.markdown(f"### Transcription:")
                            st.markdown(f"{highlight['Transcription']}")
                        if highlight["Summary"]:
                            st.markdown(f"### Summary:")
                            st.markdown(f"{highlight['Summary']}")
                        st.markdown(f"### Highlights:")
                        st.write(highlight["Highlight"])
                        st.write("---")
                        st.write("---")
            else:
                st.warning("You do not have any previous Sessions.")

        else:
            st.warning("Please login/signup to access this page.")

    elif page == "Smart Tutor":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                for user in data["users"]:
                    if user["email"] == user_info["email"]:
                        user_info = user
                        break
            if "previous_questions" not in session_state:
                session_state["previous_questions"] = []
            previous_lectures = [
                lecture["Transcription"] 
                for lecture in sorted(
                    user_info["highlights"],
                    key=lambda x: datetime.datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S"),
                    reverse=True
                )
            ]
            session_state["previous_lectures"] = previous_lectures

            st.title("Smart Tutor")
            st.write("Welcome to the Smart Tutor page.")
            st.write("Here you can ask questions and get answers from the Smart Tutor.")
            if len(session_state["previous_questions"]) > 0:
                for question in session_state["previous_questions"]:
                    st.chat_message("You", avatar="👤").write(question["question"])
                    st.chat_message("Smart Tutor", avatar="🧠").write(
                        question["answer"]
                    )

            if question := st.chat_input("Ask a question", key="question"):
                st.chat_message("You", avatar="👤").write(question)
                session_state["previous_questions"].append(
                    {"question": question, "answer": ""}
                )
                with st.spinner("Thinking..."):
                    answer = smart_tutor(
                        question=question,
                        previous_lectures=previous_lectures,
                        previous_questions=session_state["previous_questions"],
                    )
                    if answer is None:
                        answer = smart_tutor_gemini(
                            question=question,
                            previous_lectures=previous_lectures,
                            previous_questions=session_state["previous_questions"],
                        )
                    st.chat_message("Smart Tutor", avatar="🧠").write_stream(stream_data(answer))
                    session_state["previous_questions"][-1]["answer"] = answer
                st.rerun()
        else:
            st.warning("Please login/signup to access this page.")


if __name__ == "__main__":
    initialize_database()
    main()
