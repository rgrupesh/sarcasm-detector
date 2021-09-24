import streamlit as st
import tensorflow as tf
import pandas as pd
from clean import token_word


MAX_LENGTH = 25


def load_model():
    loaded_model = tf.keras.models.load_model('../models/my_model.h5')
    return loaded_model


def predict_sarcasm(text):
    x_final = pd.DataFrame({"headline": [text]})
    test_lines = token_word(x_final)
    tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_obj.fit_on_texts(test_lines)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = tf.keras.preprocessing.sequence.pad_sequences(
        test_sequences, maxlen=MAX_LENGTH, padding='post')
    model = load_model()
    pred = model.predict(test_review_pad)
    predict_sarcasm.probab = pred
    if pred[0][0] >= 0.5:
        st.image('../static/images/bazinga.jpg', width=200, caption="Shelly")
        return "It's a sarcasm!"
    else:
        return "It's not a sarcasm."


def main():
    st.title("Sarcasm detector")

    with st.form(key='sarcasm_form'):
        raw_text = st.text_area("Enter your text below:")

        submit_text = st.form_submit_button(label='Submit')

    if submit_text:

        if raw_text == "":
            st.exception(NameError("Please enter some text to test."))

        else:
            col1, col2 = st.columns(2)

            prediction = predict_sarcasm(raw_text)

            with col1:
                st.success("Your input")
                st.write(raw_text)
                st.success("prediction")
                st.write(prediction)

            with col2:
                st.success("Probability")
                st.write(float(predict_sarcasm.probab))

    st.subheader("Sample prediction:")
    st.info("You broke my laptop. Thank you!  \n-sarcastic    \n  \nHi, I’m Chandler. I make jokes when I’m uncomfortable.  \n-sarcastic")
    st.info(
        "You saved my dog.    \n-not sarcastic  \n  \nSarcasm is easy.  \n-not sarcastic")


if __name__ == "__main__":
    main()
