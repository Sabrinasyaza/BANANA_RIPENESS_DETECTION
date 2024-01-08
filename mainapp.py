# main_app.py

import streamlit as st
import mysql.connector
import numpy as np
from PIL import Image
from app import main as webcam_main
from history import main as history_main
from report import main as report_main

# Function to create a connection to the MySQL database
def connect_to_database():
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "banana_ripeness",
    }

    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

# Function to create the users table if not exists
def create_users_table(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            password VARCHAR(255) NOT NULL
        )
    """)
    connection.commit()
    cursor.close()

# Function to insert user Sign Updata into the database
def insert_user_data(connection, username, password):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
    connection.commit()
    cursor.close()

# Function to check if a username already exists in the database
def is_username_exists(connection, username):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# Function to create a Sign Upform
def registration_form(connection):
    st.header("Sign Up")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("SIGN UP"):
        if password == confirm_password:
            if not is_username_exists(connection, username):
                insert_user_data(connection, username, password)
                st.success("Sign Up successful. Please proceed to LOGIN.")
            else:
                st.error("Username already exists. Please choose a different username.")
        else:
            st.error("Passwords do not match. Please try again.")

# Function to create a LOGIN form
def login_form(connection):
    st.header("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("LOGIN"):
        if is_username_exists(connection, username):
            st.success("LOGIN successful. Welcome, {}".format(username))
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username. Please try again.")

def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logout successful")
        

# Main Streamlit application
def main():
    
    logo_path = "logo.png" 

    try:
        logo_image = Image.open(logo_path)
        st.image(logo_image, width=900) 
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Establish database connection
    connection = connect_to_database()

    # Create users table if not exists
    create_users_table(connection)

    # Sidebar navigation
    if not st.session_state.logged_in:
        page = st.sidebar.selectbox("", ["SIGN UP", "LOGIN"])
    
        if page == "SIGN UP":
            registration_form(connection)
        elif page == "LOGIN":
            login_form(connection)
    else:
        
        # Continue with the existing pages (Dashboard, Webcam)
        page = st.sidebar.selectbox("Select Page", ["Dashboard", "Webcam", "History", "Report"])
        
        #Dashboard
        if page == "Dashboard":
            st.header("Welcome, {}".format(st.session_state.username))
            
            st.markdown("---")
            st.subheader("What is Ripeness Detection System?")
            long_text = """
                <div style='text-align: justify;'>
                    In a dynamic era of agriculture, the farmers grapple with the challenges of accurately determining the
                    banana ripeness using traditional methods. This project introduces a cutting-edge solution by leveraging
                    technology, specifically employing Convolutional Neural Network (CNN) and computer vision. This
                    innovative approach promises time efficiency, heightened accuracy and minimized waste in the banana
                    harvesting process. Fruit ripeness detection is one of the technologies that can help the farmers identify and 
                    locate the banana based on the stages of the ripeness in images or videos. On the other hand, the fruit 
                    detection system involves algorithms and techniques that can recognize and differentiate the fruits 
                    based on their texture, shape, colour and size.
                </div>
                """

            st.markdown(long_text, unsafe_allow_html=True)
            st.markdown("")
            st.subheader("Did you know?")
            long_text = """
                <div style='text-align: justify;'>
                     The DARKER the banana the BETTER
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            
            long_text = """
                <div style='text-align: justify;'>
                    A fully ripe banana with dark patches on the yellow of the skin produces a substance called TNF (Tumor Necrosis Factor)
                    which has the ability to combat abnormal cells. In addition, ripe bananas also caontain higher levels of powerful 
                    antioxidants.
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            st.markdown("")
            
            st.image("bananaripenesslevel.jpg", caption="Banana Ripeness Level", width=650)
            
            st.subheader("How to Identify Banana Ripeness Level")
            long_text = """
                <div style='text-align: justify;'>
                    1. Colour Guide 
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            
            long_text = """
                <div style='text-align: justify;'>
                    2. Texture and Feel
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            
            long_text = """
                <div style='text-align: justify;'>
                    3. The aroma of the bananas  
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            
            long_text = """
                <div style='text-align: justify;'>
                    4. Taste Preferences  
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)
            
            long_text = """
                <div style='text-align: justify;'>
                    5. Weight  
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)     
            
            st.markdown("")            
                    
            st.subheader("Banana Ripeness Guide")
            long_text = """
                <div style='text-align: justify;'>
                    Bananas go through several stages of ripeness, each offering a unique taste and texture. 
                    In the green, unripe stage, bananas are firm with a starchy flavor. 
                    As they transition to a yellow hue with green tips, they begin to soften, and a subtle sweetness emerges. 
                    When the banana is half yellow, it reaches an ideal ripeness with a balanced texture and flavor.
                    As bananas progress to mostly yellow, they become very ripe, featuring a soft texture and increased sweetness. 
                    Fully yellow bananas with brown spots are overripe, providing an even sweeter taste but may be mushy. 
                    If the peel turns entirely brown, the banana is overripe and best suited for baking or smoothies.  
                </div>
                """
            st.markdown(long_text, unsafe_allow_html=True)  
            st.markdown("")   
            
            st.image("ripenesschart.jpg", caption="Banana Ripeness Chart", width=650)  
                    
            
        elif page == "Webcam":
            webcam_main()
            return
            
        elif page == "History":
            history_main()
            return
        
        elif page == "Report":
            report_main()
            return
        
        logout_button() 

    # Close the database connection when the application ends
    if connection:
        connection.close()

if __name__ == "__main__":
    main()
