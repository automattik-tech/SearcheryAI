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

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", clean_up_tokenization_spaces=False)

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
    query_embedding = model.encode([query])  # Pass query as a list (ensures 2D array)
    query_embedding = query_embedding.reshape(1, -1)  # Ensure it's a 2D array (1xN)

    similarity = cosine_similarity(query_embedding, document_embeddings)
    results = np.argsort(similarity[0])[::-1][:5]

    snippets = [(get_snippet(documents[index], query), filenames[index]) for index in results]

    return render_template('index.html', query=query, results=results, documents=documents, snippets=snippets)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answers = []

    query_embedding = model.encode([question])  # Pass question as a list (ensures 2D array)
    query_embedding = query_embedding.reshape(1, -1)  # Ensure it's a 2D array (1xN)

    similarity = cosine_similarity(query_embedding, document_embeddings)
    results = np.argsort(similarity[0])[::-1][:5]
    context = " ".join([documents[index] for index in results])

    answer_count = 3
    for _ in range(answer_count):
        result = qa_model(question=question, context=context)
        answers.append((result['answer'], result['score'], filenames[results[0]]))

    best_answers = sorted(answers, key=lambda x: x[1], reverse=True)[:answer_count]

    snippets = [(get_snippet(context, answer[0]), answer[2]) for answer in best_answers]

    return render_template('index.html', question=question, answers=best_answers, snippets=snippets)

@app.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/files', methods=['GET'])
def list_files():
    return jsonify(filenames)

@app.route('/uploaded-files')
def uploaded_files():
    return render_template('uploaded_files.html', filenames=filenames)

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        global documents, document_embeddings, filenames
        documents, filenames = load_documents_from_files(UPLOAD_FOLDER)
        document_embeddings = model.encode(documents)
        return jsonify({'success': True, 'message': f'{filename} deleted successfully'})
    return jsonify({'success': False, 'message': f'{filename} not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return url_for('static', filename='images/favicon.ico')

if __name__ == '__main__':
    app.run(debug=True)
