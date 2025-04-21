import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model #type: ignore
import base64
from transformers import TFEncoderDecoderModel, BertTokenizerFast
from keras_preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json #type: ignore
from transformers import T5Tokenizer, T5ForConditionalGeneration


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
    
@st.cache_resource
def load_lstm_models():
    encoder_model = load_model("encoder_model.keras")
    decoder_model = load_model("decoder_model.keras")
    # Load tokenizer from file
    with open("x_tokenizer.json", "r") as json_file:
        tokenizer_json = json_file.read()
        x_tokenizer = tokenizer_from_json(tokenizer_json)
    # Load tokenizer from file
    with open("y_tokenizer.json", "r") as json_file:
        tokenizer_json = json_file.read()
        y_tokenizer = tokenizer_from_json(tokenizer_json)
    return encoder_model,decoder_model,x_tokenizer,y_tokenizer


@st.cache_resource
def load_bert_models():
    model = TFEncoderDecoderModel.from_pretrained("trained_bert2bert.h5")
    tokenizer = BertTokenizerFast.from_pretrained("tokenizer/")
    return model, tokenizer

@st.cache_resource
def load_t5_models():
    model = T5ForConditionalGeneration.from_pretrained('t5_model.pt')
    tokenizer = T5Tokenizer.from_pretrained('t5_tokenizer.pt')
    return model, tokenizer



def predict_lstm(input_text):
    encoder_model,decoder_model,x_tokenizer,y_tokenizer=load_lstm_models()
    x = np.array([input_text])
    reverse_target_word_index = y_tokenizer.index_word 
    reverse_source_word_index = x_tokenizer.index_word 
    target_word_index = y_tokenizer.word_index 
    max_len_text = 80 
    max_len_summary = 10
    def decode_sequence(input_seq):
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = target_word_index['sostok']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            e_h, e_c = h, c
        return decoded_sentence
   
    def seq2summary(input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
                newString=newString+reverse_target_word_index[i]+' '
        return newString
    def seq2text(input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+reverse_source_word_index[i]+' '
        return newString
    x = pad_sequences(x_tokenizer.texts_to_sequences(x), maxlen=max_len_text, padding='post')
    return decode_sequence(x.reshape(1, max_len_text))

def predict_bert(input_text):
    model, tokenizer = load_bert_models()
    summary, _ = generate_summary(
        model, 
        tokenizer, 
        input_text  
    )
    return summary


def generate_summary(model, tokenizer, input_text):
    input_seq = tokenizer(input_text, return_tensors='tf', max_length=512, truncation=True)
    outputs = model.generate(
        input_ids=input_seq['input_ids'], 
        attention_mask=input_seq['attention_mask']
    )
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_str, input_text
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

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")

def t5_generate_summary(text, model, tokenizer, max_length=150):
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs.to(model.device),
                                 max_length=max_length,
                                 min_length=30,
                                 length_penalty=2.0,
                                 num_beams=4,
                                 early_stopping=True)

    # Decode the summary tokens to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


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

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None





    

def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

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
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def main(json_file_path="data.json"):
    st.sidebar.title("Abstractive Text Summarizaton")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Abstractive Text Summarizaton"),
        key="Abstractive Text Summarizaton",
    )

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

    elif page == "Abstractive Text Summarizaton":
        if session_state.get("logged_in"):
            st.title("Abstractive Text Summarization")
            text = st.text_area("Enter Your Text")
            options =['LSTM','BERT','T5','COMPARISON']
            model = st.radio('Select an option',options)
            if st.button("submit") and text is not None:
                if model == 'LSTM':
                    summary = predict_lstm(text)
                    st.write("Summary:", summary)        
                elif model =='BERT':
                    # mod ='trained_bert2bert_h5\kaggle\working\trained_bert2bert.h5'
                    loaded_model_b = TFEncoderDecoderModel.from_pretrained("trained_bert2bert.h5")
                    loaded_tokenizer_b = BertTokenizerFast.from_pretrained("tokenizer/")
                    prediction = predict_bert(loaded_model_b, loaded_tokenizer_b, text)
                    st.write("Summary:", prediction)
                elif model=='T5':
                    loaded_model_t5 = T5ForConditionalGeneration.from_pretrained('t5_model.pt')
                    loaded_tokenizer_t5 = T5Tokenizer.from_pretrained('t5_tokenizer.pt')
                    prediction = t5_generate_summary(text,loaded_model_t5,loaded_tokenizer_t5)
                    st.write("Summary:", prediction)
                else:
                    summary = predict_lstm(text)
                    st.write("LSTM Summary:", summary)
                    loaded_model_b = TFEncoderDecoderModel.from_pretrained("trained_bert2bert.h5")
                    loaded_tokenizer_b = BertTokenizerFast.from_pretrained("tokenizer/")
                    prediction = predict_bert(loaded_model_b, loaded_tokenizer_b, text)
                    st.write("BERT Summary:", prediction)
                    loaded_model_t5 = T5ForConditionalGeneration.from_pretrained('t5_model.pt')
                    loaded_tokenizer_t5 = T5Tokenizer.from_pretrained('t5_tokenizer.pt')
                    prediction1 = t5_generate_summary(text,loaded_model_t5,loaded_tokenizer_t5)
                    st.write("T5 Summary:", prediction1)
        else:
            st.warning("Please login/signup to use App!!")

      




if __name__ == "__main__":
    initialize_database()
    main()
