import os
import re
import ollama

def readtextfiles(path):
    text_contents = {}
    directory = os.path.join(path)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Try different encodings in order of likelihood
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as file:
                        content = file.read()
                    text_contents[filename] = content
                    break  # Successfully read the file, move to next file
                except UnicodeDecodeError:
                    continue  # Try next encoding if this one fails
            else:  # This runs if no encoding worked
                print(f"Warning: Could not read {filename} with any of the attempted encodings")

    return text_contents

def chunksplitter(text, chunk_size=100):
  words = re.findall(r'\S+', text)

  chunks = []
  current_chunk = []
  word_count = 0

  for word in words:
    current_chunk.append(word)
    word_count += 1

    if word_count >= chunk_size:
      chunks.append(' '.join(current_chunk))
      current_chunk = []
      word_count = 0

  if current_chunk:
    chunks.append(' '.join(current_chunk))

  return chunks

def getembedding(chunks):
  embeds = ollama.embed(model="nomic-embed-text", input=chunks)
  return embeds.get('embeddings', [])