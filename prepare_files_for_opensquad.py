import unicodedata
import numpy
import sqlite3
import os
import json
import pickle

DB_PATH = 'wikipedia/docs.db'

##========== the function and class below come from https://github.com/facebookresearch/DrQA (please check their license)


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path or DB_PATH
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

PROCESS_DB = DocDB()
all_docs_ids = PROCESS_DB.get_doc_ids()

# Passage Chunking
DO_CHUNKING = True
PASSAGE_LENGTH = 100
DOC_STRIDE = 50

# this function cuts articles into pieces of size "chunk_length" with a sliding window of "stride" words
def chunk_article(article: str, chunk_length: int, stride: int) -> ([str], [int]):    
    article_splitted = article.split(" ")
    num_words = len(article_splitted)
    if num_words > chunk_length :
        steps = int(numpy.ceil((num_words - chunk_length)/stride))
        beg = [i*stride for i in range(steps + 1)]
        end = [i*stride + chunk_length  for i in range(steps)] + [num_words]
        chunks = [" ".join(article_splitted[beg[i]:end[i]]) for i in range(len(end))]
        offsets = beg
    else:
        chunks = [article]
        offsets = [0]
    return chunks, offsets

print("======= Preparing passages from wikipedia dump ==========")

chunks_array = []
for i,article in enumerate(all_docs_ids):
    if i%100000 == 0:
        print("Processing artcle number ", i)
        
    doc_text = PROCESS_DB.get_doc_text(article)
    passages, offsets = chunk_article(doc_text, PASSAGE_LENGTH, DOC_STRIDE)
    for j, part in enumerate(passages):
        chunks_array.append({
          "id": article+'_'+str(j),
          "contents": part
        })

if not os.path.exists('formatted_open_squad'):
    os.makedirs('formatted_open_squad')
    
if not os.path.exists('formatted_open_squad/paragraphs_json'):
    os.makedirs('formatted_open_squad/paragraphs_json')

if not os.path.exists('formatted_open_squad/indexes'):
    os.makedirs('formatted_open_squad/indexes')

with open("formatted_open_squad/paragraphs_json/documents.json",'w') as f1:
    json.dump(chunks_array,f1)   
    
print("======== formatting squad v1.1 dev set =============")

with open("SQuAD_1_1/dev-v1.1.json","r") as f1:
    squad1 = json.load(f1)

all_questions = []
all_answers = []
all_ids = []
for article in squad1['data']:
    print(article['title'])
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            all_questions.append(qa['question'])
            all_answers.append(qa['answers'][0]['text'])
            all_ids.append(qa['id'])


squad1_for_orqa = {}
squad1_for_orqa['questions'] = all_questions
squad1_for_orqa['answers'] = all_answers
squad1_for_orqa['ids'] = all_ids

with open('formatted_open_squad/open_squad.pkl','wb') as f1:
    pickle.dump(squad1_for_orqa,f1)