# app.py
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bookreco.db'
db = SQLAlchemy(app)

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    genre = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Book {self.title}>'

# Initialize the database
with app.app_context():
    db.create_all()

    # Add some sample books if the database is empty
    if Book.query.count() == 0:
        sample_books = [
            Book(title="The Great Gatsby", author="F. Scott Fitzgerald", genre="Classic", description="A story of decadence and excess in Jazz Age America"),
            Book(title="To Kill a Mockingbird", author="Harper Lee", genre="Classic", description="A novel about racial injustice and loss of innocence in the American South"),
            Book(title="1984", author="George Orwell", genre="Science Fiction", description="A dystopian novel set in a totalitarian society"),
            Book(title="The Hobbit", author="J.R.R. Tolkien", genre="Fantasy", description="A fantasy novel about a hobbit's journey to win a share of treasure guarded by a dragon"),
            Book(title="Pride and Prejudice", author="Jane Austen", genre="Romance", description="A romantic novel of manners set in Georgian England")
        ]
        db.session.add_all(sample_books)
        db.session.commit()

@app.route('/')
def home():
    books = Book.query.all()
    return render_template('index.html', books=books)

@app.route('/recommend', methods=['POST'])
def recommend():
    book_id = request.form.get('book_id')
    book = Book.query.get(book_id)
    
    if not book:
        return jsonify({'error': 'Book not found'}), 404

    # Get all books
    books = Book.query.all()
    
    # Create a list of all book descriptions
    descriptions = [b.description for b in books]
    
    # Create the count matrix
    count = CountVectorizer().fit_transform(descriptions)
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count, count)
    
    # Get the index of the book that matches the title
    indices = {b.id: i for i, b in enumerate(books)}
    idx = indices[book.id]
    
    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 5 most similar books
    sim_scores = sim_scores[1:6]
    
    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    
    # Return the top 5 most similar books
    recommended_books = [books[i] for i in book_indices]
    
    return jsonify({
        'recommendations': [
            {
                'id': book.id,
                'title': book.title,
                'author': book.author,
                'genre': book.genre
            } for book in recommended_books
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)