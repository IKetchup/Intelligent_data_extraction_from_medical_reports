from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import json

#split_percent = round(len(train['annotations']) * 0.7)

corpus = "annotated_text.json"

# Load the annotated corpus
with open(corpus, encoding='utf8') as f:
    reports = json.load(f)

# Convert the tagged data to a format understood by SpaCy
TRAIN_DATA  = []
for content, entities in reports["annotations"]:
    if len(entities["entities"]) > 0:
        TRAIN_DATA.append(([content,entities]))

nlp = spacy.blank("fr") # load a new spacy model

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA[0:300]): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            print(msg)
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)
db.to_disk("./new_train.spacy") # save the docbin object

nlp = spacy.blank("fr") # load a new spacy model

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA[300:-1]): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            print(msg)

        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)
db.to_disk("./new_dev.spacy") # save the docbin object

