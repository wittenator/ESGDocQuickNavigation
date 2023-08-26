#! python3

from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session

from typing import TypedDict

from chunker import extract_text_from_pdf, get_splits

class Chunk(TypedDict):
    chunk: str
    chunk_location_metadadata: str


# read env file
from dotenv import load_dotenv
import os
load_dotenv()

engine = create_engine(f'postgresql+psycopg://localhost:5432/{os.getenv("POSTGRES_DB")}?user={os.getenv("POSTGRES_USER")}&password={os.getenv("POSTGRES_PASSWORD")}')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    chunk = mapped_column(Text)
    embedding = mapped_column(Vector(1024))
    chunk_location_metadadata = mapped_column(Text)

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
session = Session(engine)

def embedd_chunks(chunks: list[Chunk]):
    model = SentenceTransformer('BAAI/bge-large-zh')
    # model = SentenceTransformer('BAAI/bge-large-en')
    embeddings = model.encode([chunk['chunk'] for chunk in chunks])
    documents = [dict(chunk=chunks[i]['chunk'], chunk_location_metadadata=chunks[i]['chunk_location_metadadata'], embedding=embedding) for i, embedding in enumerate(embeddings)]
    session.execute(insert(Document), documents)
    session.commit()

def query_chunks(query: str):
    model = SentenceTransformer('BAAI/bge-large-zh')
    # model = SentenceTransformer('BAAI/bge-large-en')
    embedding = model.encode(query)
    neighbors = session.scalars(select(Document).order_by(Document.embedding.cosine_distance(embedding)).limit(5))
    return neighbors

if __name__ == '__main__':

    # cli interface for embedding chunks
    import argparse
    parser = argparse.ArgumentParser(description='Embed chunks')
    # argument for pdf document as file path
    parser.add_argument('--pdf', type=str, help='pdf file path', required=False)
    parser.add_argument('--query', type=str, help='query', required=False)

    args = parser.parse_args()

    if args.pdf:
        # delete last document from db
        session.query(Document).delete()
        session.commit()

        # extract chunks from pdf
        sample, metadata = extract_text_from_pdf(args.pdf)
        texts = get_splits(200, 20, sample, metadata)

        embedd_chunks([Chunk(chunk=text.page_content, chunk_location_metadadata=text.metadata["page"]) for text in texts])
        docs = query_chunks(args.query)
        for neighbor in docs:
            print(neighbor.chunk)
            print(f"page: {neighbor.chunk_location_metadadata}")
            
    elif args.query:
        docs = query_chunks(args.query)
        for neighbor in docs:
            print(neighbor.chunk)



