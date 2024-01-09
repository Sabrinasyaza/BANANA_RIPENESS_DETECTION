import streamlit as st
import pandas as pd
import mysql.connector
from PIL import Image
from io import BytesIO

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='banana_ripeness',
    port='3306'
)
cursor = connection.cursor()

def fetch_all_data(connection):
    # Fetch all data from the database
    query = "SELECT id, ripeness FROM banana_data"
    cursor.execute(query)
    result = cursor.fetchall()
    return result

def fetch_image_by_id(connection, selected_id):
    try:
        # Fetch image data based on the selected ID
        query = "SELECT picture FROM banana_data WHERE id = %s"
        cursor.execute(query, (selected_id,))
        result = cursor.fetchone()

        if result:
            return result[0]
        else:
            st.warning("No image found for the selected ID.")
            return None

    except Exception as e:
        st.error(f"Error fetching image: {e}")
        import traceback

def display_image(img_bytes):
    try:
        # Convert the image bytes to a PIL Image
        img = Image.open(BytesIO(img_bytes))
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def history_main():
    st.header("Display Captured Images and Data")
    
    # Fetch all data from the database
    all_data = fetch_all_data(connection)
    
    # Display IDs and let the user select one
    selected_id = st.selectbox("Select ID:", [data[0] for data in all_data])

    # Display the selected ID's data
    selected_data = [data for data in all_data if data[0] == selected_id][0]
    st.write(f"Selected ID: {selected_data[0]}")
    st.write(f"Banana Ripeness: {selected_data[1]}")

    # Fetch and display the image based on the selected ID
    img_bytes = fetch_image_by_id(connection, selected_id)
    if img_bytes:
        display_image(img_bytes)
    else:
        st.write("No image found for the selected ID.")

if __name__ == "__main__":
    history_main()
