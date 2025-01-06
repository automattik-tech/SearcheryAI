from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import PyPDF2
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploaded_files')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_summarization_model(): 
    return pipeline("summarization", model="facebook/bart-large-cnn")

def load_documents_from_files(upload_folder):
    documents = []
    filenames = []
    for filename in os.listdir(upload_folder):
        if allowed_file(filename):
            file_path = os.path.join(upload_folder, filename)
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'pdf':
                documents.append(extract_text_from_pdf(file_path))
            elif ext == 'docx':
                documents.append(extract_text_from_docx(file_path))
            elif ext == 'txt':
                documents.append(extract_text_from_txt(file_path))
            elif ext == 'png':
                documents.append(f'<img src="/files/{filename}" alt="{filename}" class="header-image">')
            filenames.append(filename)
    return documents, filenames

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text.strip()

def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def get_snippet(document, query, length=50):
    query_lower = query.lower()
    start_index = document.lower().find(query_lower)
    if start_index == -1:
        return None

    start = max(start_index - length, 0)
    end = start_index + len(query) + length
    snippet = document[start:end].replace('\n', ' ')
    snippet = snippet.replace(query, f'<strong>{query}</strong>')

    if start > 0:
        snippet = '...' + snippet
    if end < len(document):
        snippet += '...'
    
    return snippet

documents, filenames = load_documents_from_files(UPLOAD_FOLDER)
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents)

qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", clean_up_tokenization_spaces=False)

@app.route('/')
def index():
    return render_template('index.html', documents=documents)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global documents, document_embeddings, filenames
        documents, filenames = load_documents_from_files(UPLOAD_FOLDER)
        document_embeddings = model.encode(documents)
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, document_embeddings)
    results = np.argsort(similarity[0])[::-1][:5]

    snippets = [(get_snippet(documents[index], query), filenames[index]) for index in results]

    return render_template('index.html', query=query, results=results, documents=documents, snippets=snippets)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answers = []

    # Find the most relevant documents
    query_embedding = model.encode([question])
    similarity = cosine_similarity(query_embedding, document_embeddings)
    results = np.argsort(similarity[0])[::-1][:5]

    # Combine relevant document contexts
    context = " ".join([documents[index] for index in results if documents[index].strip()])

    if not context:
        # Handle case where no relevant documents are found
        return render_template(
            'index.html',
            question=question,
            answers=[("No relevant documents found to answer your question.", 0, "N/A")],
            snippets=[]
        )

    # Summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Use QA model to extract potential answers
    for index in results:
        doc_context = documents[index]
        if doc_context.strip():
            result = qa_model(question=question, context=doc_context)
            answers.append((result['answer'], result['score'], filenames[index]))

    # Summarize the combined answers
    combined_context = " ".join([ans[0] for ans in answers if ans[0].strip()])
    if combined_context.strip():
        summarized_answer = summarizer(combined_context, max_length=100, min_length=30, do_sample=False)
        summary_text = summarized_answer[0]['summary_text']
    else:
        summary_text = "Unable to generate a summarized answer from the available context."

    # Prepare snippets and filenames for display
    snippets = [(get_snippet(context, summary_text), filenames[results[0]])]

    return render_template(
        'index.html',
        question=question,
        answers=[(summary_text, 1.0, filenames[results[0]])],
        snippets=snippets
    )


@app.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/files', methods=['GET'])
def list_files():
    return jsonify(filenames)

@app.route('/uploaded-files')
def uploaded_files():
    return render_template('uploaded_files.html', filenames=filenames)


if __name__ == '__main__':
    app.run(debug=True)
