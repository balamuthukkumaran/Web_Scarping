import requests
from bs4 import BeautifulSoup
import time
from flask import Flask, render_template, request
import csv
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Check robots.txt before crawling
url = "https://pureportal.coventry.ac.uk/robots.txt"
response = requests.get(url)
robots = response.text

if "User-agent: *\nDisallow: /" in robots:
    print("This website disallows web crawlers. Aborting...")
    exit()

delay = 1  

# URLs to crawl
urls = [
    "https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/",
    "https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/?page=1",
    "https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/?page=2",
    "https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/?page=3",
    "https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/?page=4"]

papers = []

for url in urls:
    # Wait for the delay before sending the request
    time.sleep(delay)
    
    # Send the request
    response = requests.get(url)
    
    # Check the status code of the response
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}. Skipping...")
        continue
    
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all(class_="result-container")

    for result in results:
        if result.find(class_="link person"):
            html = result
            span1 = []
            for span in result.select('h3.title ~ span:not([class])'):
                if span.find_previous_sibling() is None or not span.find_previous_sibling().has_attr('rel='):
                    span1.append(span.text.strip())
                  
            authors = result.find_all(class_="link person")

        paper = {}
        title = result.find(class_="title")
        if title is not None:
            title_link = title.find(class_="link")
            if title_link is not None:
                paper["title"] = title.get_text().strip()
                paper["title_link"] = title_link.get('href')
         
        year = result.find(class_="date")
        if year is not None:
            paper["year"] = year.get_text().strip()
            
        authors = result.find_all(class_="link person")
        if authors:
            author_links = [author.get('href') for author in authors if author.has_attr('href')]
            author_names = [author.get_text().strip() for author in authors]
            for i in span1:
                author_names.append(i)
            if author_links:
                paper["authors"] = author_names
                paper["author_links"] = author_links
                author_links_list = paper["author_links"]
                author_links_str = '"' + '", "'.join(author_links_list) + '"'
                paper["author_links"] = author_links_str
                
        # Add the paper dictionary to the list of papers
        if "authors" in paper:
            papers.append(paper)

# Preprocess the data
for paper in papers:
    
    # Remove any non-alphanumeric characters from the title
    paper["title"] = ''.join(c for c in paper["title"] if c.isalnum() or c.isspace())
    
    # Convert the author names to lowercase
    if "authors" in paper:
        paper["authors"] = [author.lower() for author in paper["authors"]]     

# Save the data in a CSV file
import csv
filename = "papers.csv"
with open(filename, mode="w", newline="") as csvfile:
    fieldnames = ["title", "title_link", "year", "authors", "author_links"]
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for paper in papers:
        authors_formatted = ", ".join(paper["authors"])
        paper["authors"] = authors_formatted
        writer.writerow(paper)            

app = Flask(__name__)

# Load the papers from the CSV file
filename = "papers.csv"
papers = []
with open(filename, mode="r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        papers.append(row)

# Define the stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocess the text data for each paper
for paper in papers:
    
    # Tokenize the title and abstract
    title_tokens = word_tokenize(paper["title"].lower())
    year_tokens = word_tokenize(paper["year"])
    authors_tokens = word_tokenize(paper["authors"].lower())
    
    # Remove stop words and stem the remaining words
    title_stems = [stemmer.stem(word) for word in title_tokens if word not in stop_words]
    year_stems = [stemmer.stem(word) for word in year_tokens if word not in stop_words]
    authors_stems = [stemmer.stem(word) for word in authors_tokens if word not in stop_words]
    
    # Combine the title and abstract stems
    all_stems = title_stems + year_stems + authors_stems 

    # Create a frequency dictionary for the stems
    freq_dict = defaultdict(int)
    for stem in all_stems:
        freq_dict[stem] += 1
    
    # Calculate the TF-IDF score for each stem and store it in the paper dictionary
    num_docs = len(papers)
    for stem, freq in freq_dict.items():

        # Calculate the term frequency (TF) for the stem in this paper
        tf = freq / len(all_stems)
        
        # Calculate the inverse document frequency (IDF) for the stem across all papers
        num_docs_with_stem = sum(1 for p in papers if stem in p)
        idf = math.log(num_docs / (1 + num_docs_with_stem))
        
        # Calculate the TF-IDF score for the stem in this paper
        tf_idf = tf * idf
        
        # Store the TF-IDF score for the stem in the paper dictionary
        if "tf_idf" not in paper:
            paper["tf_idf"] = {}
        paper["tf_idf"][stem] = tf_idf
 
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/search", methods=['GET', 'POST'])

def search():
    # Get the search query from the user
    query = request.form.get("query")
        
    # Tokenize and preprocess the search query
    query_tokens = [stemmer.stem(token) for token in word_tokenize(query.lower()) if token not in stop_words]
    
    # Compute the scores for each paper based on the search query
    scores = defaultdict(float)
    for i, paper in enumerate(papers):
        title_tokens = [stemmer.stem(token) for token in word_tokenize(paper["title"].lower()) if token not in stop_words]
        author_tokens = [stemmer.stem(token) for token in word_tokenize(paper["authors"].lower()) if token not in stop_words]
        all_tokens = title_tokens + author_tokens
    
        for token in query_tokens:
            if token in all_tokens:
                scores[i] += 1
    
    # Sort the papers by score and return the top 10
    top_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get the paper information for the top 10 papers
    results = []
    for paper_id, score in top_papers:
        paper = papers[paper_id]
        results.append({
            "title": paper["title"],
            "title_link": paper["title_link"],
            "year": paper["year"],
            "authors": paper["authors"],
            "author_links": paper["author_links"],
        })
    return render_template("results.html", query=query, papers=results)

if __name__ == "__main__":
    app.run(debug=True)

