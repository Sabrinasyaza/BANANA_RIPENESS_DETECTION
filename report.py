import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector

# Connect to MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="banana_ripeness"
)
cursor = connection.cursor()

def fetch_ripeness_counts(connection):
    # Fetch ripeness counts from the database
    query = "SELECT ripeness, COUNT(*) AS count FROM banana_data GROUP BY ripeness"
    cursor.execute(query)
    result = cursor.fetchall()
    return result

def plot_ripeness_counts(ripeness_counts):
    # Convert result to a DataFrame for easier plotting
    df = pd.DataFrame(ripeness_counts, columns=['Ripeness', 'Count'])

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Ripeness'], df['Count'], color=['brown', 'yellow', 'green'])
    ax.set_xlabel('Banana Ripeness')
    ax.set_ylabel('Count')
    ax.set_title('Total Count of Banana Ripeness Results')

    # Show the plot within Streamlit
    st.pyplot(fig)

def main():
    st.header("Banana Ripeness Report")

    # Fetch ripeness counts from the database
    ripeness_counts = fetch_ripeness_counts(connection)

    # Display the ripeness counts in a DataFrame
    st.write("Ripeness Counts:")
    st.dataframe(pd.DataFrame(ripeness_counts, columns=['Ripeness', 'Count']))

    # Plot the bar chart
    st.write("Bar Chart of Ripeness Counts:")
    plot_ripeness_counts(ripeness_counts)

if __name__ == "__main__":
    main()
