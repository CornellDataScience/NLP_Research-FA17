from lxml import html
import requests
import re
import pandas as pd


def book(title, author, link):
    return {
        'title': title,
        'author': author,
        'link': link
    }


books = [
    book("THE TEN BOOKS ON ARCHITECTURE",
         "Vitruvius",
         "https://www.gutenberg.org/files/20239/20239-h/20239-h.htm"),
    book("THE STORY OF THE LIVING MACHINE",
         "H.W. CONN",
         "https://www.gutenberg.org/files/16487/16487-h/16487-h.htm"),
    book("THE WORKS OF EDGAR ALLAN POE",
         "Edgar Allan Poe",
         "https://www.gutenberg.org/files/2147/2147-h/2147-h.htm"),
    book("A TALE OF TWO CITIES",
         "Charles Dickens",
         "https://www.gutenberg.org/files/98/98-h/98-h.htm"),
    book("The Tragedie of Hamlet",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/2265/pg2265.html"),
    book("ANDERSEN’S FAIRY TALES",
         "Hans Christian Anderson",
         "https://www.gutenberg.org/files/1597/1597-h/1597-h.htm"),
    book("Stories the Iroquois Tell Their Children",
         "Mabel Powers",
         "https://www.gutenberg.org/files/22096/22096-h/22096-h.htm")
]

frames = []

for b in books:
    print(b['title'])
    page = requests.get(b['link'])
    tree = html.fromstring(page.content)

    pars = tree.xpath('//p/text()')
    b['text'] = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
                         re.sub("<span.*>", '', "\n".join(pars)))
    frame = pd.DataFrame(b)
    print(frame.shape)
    frames.append(frame.iloc[1:, :])

pd.concat(frames).reset_index().to_json("./text_sentences.json")
