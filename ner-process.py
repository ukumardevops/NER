import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer, BertModel
from datetime import datetime
import faiss
import streamlit as st
import pandas as pd

model_name = 'all-MiniLM-L6-v2'
# Load the Sentence Transformer model for embedding generation
sentence_model = SentenceTransformer(model_name)
model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Preprocess the comment
def preprocess_comment(comment):
    return comment.replace('-', ' ').replace(':', '').strip()


# Extract entity using bert model
def extract_entity(comment):
    tokenized_comment = tokenizer(comment, return_tensors='pt')
    outputs = model(**tokenized_comment).logits
    predictions = outputs.argmax(dim=2)
    
    entities = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions[0]]

    return ' '.join(entities)


# Validate and match entity with known titles
def validate_entity(extracted_entity, known_titles):
    title_embeddings = sentence_model.encode(known_titles)
    entity_embedding = sentence_model.encode([extracted_entity])
    index = faiss.IndexFlatL2(title_embeddings.shape[1])
    index.add(title_embeddings)
    distances, indices = index.search(entity_embedding, 1)
    best_match_index = indices[0][0]
    return known_titles[best_match_index] if distances[0][0] < 0.5 else None


# Extract date if available
def extract_date(comment):
    date_pattern = r'\d{1,2}(st|nd|rd|th)? \w+ \d{4}'
    match = re.search(date_pattern, comment)
    return datetime.strptime(match.group(), '%dth %B %Y') if match else None


# Find full title from the comment
def find_full_title(comment):
    preprocessed_comment = preprocess_comment(comment)
    extracted_entity = extract_entity(preprocessed_comment)
    known_titles = [
        'Spider-Man: Into the Spider-Verse', 'Spider-Man: Far From Home',
        'Spider-Man: No Way Home', 'Spider-Man: Across the Spider-Verse'
    ]
    validated_entity = validate_entity(extracted_entity, known_titles)
    release_date = extract_date(comment)
    
    if validated_entity:
        if release_date:
            entity = validated_entity.lower().replace(" ", "-")
            database = {
                'spider-man': [
                    {'title': 'Spider-Man: Into the Spider-Verse', 'release_date': 'December 2018'},
                    {'title': 'Spider-Man: Far From Home', 'release_date': 'July 2019'},
                    {'title': 'Spider-Man: No Way Home', 'release_date': 'December 2021'},
                    {'title': 'Spider-Man: Across the Spider-Verse', 'release_date': 'June 2023'}
                ]
            }
            for entry in database.get(entity, []):
                if release_date.strftime('%B %Y') in entry['release_date']:
                    return entry['title']
            return "Title not found with provided release date"
        else:
            return validated_entity

    return "Entity or date not found"



if __name__ == "__main__":
    ## Streamlit UI
    st.set_page_config(page_title="NER Application")
    st.header("Find Movie, Game Names")

    input = st.text_input("Input: ",key="input")
    uploaded_file=st.file_uploader("Choose a file to upload", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.table(df)

    submit = st.button("Search")
    if submit:
        with st.spinner('Searching...'):
            response = find_full_title(input)
            st.subheader("The Response is")
            st.write(response)

# Example comment
#comment = "Did you watch spider-man? posted on 10th June 2023."
#full_title = find_full_title(comment)
#print(full_title)

#streamlit run ner-process.py