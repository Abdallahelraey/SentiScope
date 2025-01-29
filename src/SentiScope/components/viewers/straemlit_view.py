import streamlit as st
import requests
import pandas as pd
import io

# Configuration
API_BASE_URL = "http://127.0.0.1:8000/api/v1/sentiment"

# Page Configuration
st.set_page_config(page_title="SentiScope Sentiment Analyzer", page_icon=":speech_balloon:")

# Title and Description
st.title("🔍 SentiScope Sentiment Analysis")
st.write("Analyze sentiment of texts or CSV files")

# Tabs for different input methods
tab1, tab2 = st.tabs(["Text Input", "CSV Input"])

# Text Input Tab
with tab1:
    st.subheader("Analyze Texts")
    input_texts = st.text_area(
        "Enter texts to analyze (one per line):", 
        height=200, 
        placeholder="Enter your texts here. Each line will be analyzed separately."
    )

    if st.button("Analyze Texts"):
        if input_texts:
            # Clean up texts (remove empty lines)
            cleaned_texts = [text.strip() for text in input_texts.split("\n") if text.strip()]
            
            if cleaned_texts:
                try:
                    # Send prediction request
                    response = requests.post(f"{API_BASE_URL}/predict", json={"texts": cleaned_texts})
                    response.raise_for_status()
                    results = response.json()
                    
                    # Display Results
                    st.subheader("Sentiment Analysis Results")
                    
                    for text, prediction in zip(results['input_texts'], results['predictions']):
                        # Determine color based on sentiment
                        if prediction == 'good':
                            color = 'green'
                            emoji = '😊'
                        elif prediction == 'bad':
                            color = 'red'
                            emoji = '😞'
                        elif prediction == 'neutral':
                            color = 'blue'
                            emoji = '😐'
                        else:
                            color = 'black'
                            emoji = '❓'

                        
                        # Display result
                        st.markdown(f"""
                        **Text:** {text}
                        **Sentiment:** <span style='color:{color}'>{prediction.capitalize()} {emoji}</span>
                        """, unsafe_allow_html=True)
                    
                    # Summary
                    st.subheader("Sentiment Summary")
                    predictions_summary = {}
                    for pred in results['predictions']:
                        predictions_summary[pred] = predictions_summary.get(pred, 0) + 1
                    
                    for sentiment, count in predictions_summary.items():
                        st.write(f"{sentiment.capitalize()} Texts: {count}")
                
                except requests.RequestException as e:
                    st.error(f"Error connecting to sentiment analysis service: {e}")
            else:
                st.warning("Please enter at least one non-empty text.")


with tab2:
    st.subheader("Analyze CSV")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if not text_columns:
                st.error("CSV file must contain at least one text column")
            else:
                selected_text_column = st.selectbox(
                    "Select text column to analyze", 
                    text_columns,
                    help="Choose the column containing text to analyze"
                )

                if st.button("Analyze CSV Content"):
                    try:
                        # Prepare API request
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(
                            f"{API_BASE_URL}/predict_csv",
                            files=files,
                            data={'text_column': selected_text_column}
                        )
                        
                        # Handle API response
                        response.raise_for_status()
                        results = response.json()
                        
                        if 'dataframe' not in results:
                            st.error("Unexpected response format from API")
                            st.stop()  # Changed from return

                        results_df = pd.DataFrame(results['dataframe'])
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Show data preview
                        st.dataframe(results_df.head(), use_container_width=True)
                        
                        # Show sentiment distribution
                        st.subheader("Sentiment Distribution")
                        if 'predicted_sentiment' in results_df.columns:
                            sentiment_counts = results_df['predicted_sentiment'].value_counts()
                            st.bar_chart(sentiment_counts)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Full Results",
                                data=csv,
                                file_name='sentiment_analysis_results.csv',
                                mime='text/csv',
                            )
                        else:
                            st.error("Prediction column missing in response")

                    except requests.exceptions.HTTPError as e:
                        error_detail = e.response.json().get('detail', str(e))
                        st.error(f"Analysis failed: {error_detail}")
                    except requests.RequestException as e:
                        st.error(f"Connection error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        
        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty or invalid CSV")
        except UnicodeDecodeError:
            st.error("File encoding error - please upload UTF-8 encoded CSV")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by SentiScope Sentiment Analysis Service")