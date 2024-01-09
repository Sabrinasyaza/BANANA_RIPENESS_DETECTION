import streamlit as st
import mysql.connector
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Function to connect to the MySQL database
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

# Function to insert data into the database
def insert_data(connection, ripeness, img_bytes):
    if connection:
        try:
            cursor = connection.cursor()
            query = "INSERT INTO banana_data (ripeness, picture) VALUES (%s, %s)"
            values = (ripeness, img_bytes)
            cursor.execute(query, values)
            connection.commit()
            st.success("Data inserted successfully!")
        except mysql.connector.Error as err:
            st.error(f"Error: {err}")

def detect_ripeness(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for unripe, ripe, and overripe bananas (you may need to adjust these)
    unripe_lower = np.array([30, 50, 50])
    unripe_upper = np.array([60, 255, 255])

    ripe_lower = np.array([15, 100, 50])
    ripe_upper = np.array([30, 255, 255])

    overripe_lower = np.array([0, 50, 50])
    overripe_upper = np.array([15, 255, 255])

    # Create masks for each ripeness level
    unripe_mask = cv2.inRange(hsv, unripe_lower, unripe_upper)
    ripe_mask = cv2.inRange(hsv, ripe_lower, ripe_upper)
    overripe_mask = cv2.inRange(hsv, overripe_lower, overripe_upper)

    # Calculate the percentage of each ripeness level in the image
    total_pixels = image.size / 3  # Assuming a 3-channel image
    unripe_percentage = np.sum(unripe_mask) / total_pixels
    ripe_percentage = np.sum(ripe_mask) / total_pixels
    overripe_percentage = np.sum(overripe_mask) / total_pixels

    # Determine the ripeness based on the highest percentage
    max_percentage = max(unripe_percentage, ripe_percentage, overripe_percentage)
    if max_percentage == unripe_percentage:
        return "Unripe"
    elif max_percentage == ripe_percentage:
        return "Ripe"
    else:
        return "Overripe"
    
def webcam_main():

    connection = connect_to_database()
    st.header("Ripeness Detection System")
    video_capture = cv2.VideoCapture(0)

    if st.button("Capture Image"):
        ret, frame = video_capture.read()
        st.image(frame, channels="BGR", use_column_width=True)

        ripeness = detect_ripeness(frame)
        st.write(f"Banana Ripeness: {ripeness}")

        # Convert the image to bytes
        _, img_encoded = cv2.imencode(".png", frame)
        img_bytes = img_encoded.tobytes()

        # Insert data into the database
        insert_data(connection, ripeness, img_bytes)

    # Release the webcam feed when the image is captured
    video_capture.release()

    # Close the database connection after the loop completes
    with connect_to_database() as connection:
        if connection:
            connection.close()

if __name__ == "__main__":
    webcam_main()
