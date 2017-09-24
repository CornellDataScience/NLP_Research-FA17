# processing_master.py
# Master file with methods to process data
# All methods take input of type 'pandas.core.series.Series'

import re

def lower_case(data):
    lower_case_map = {}
    for comment in data:
        lower_case_map[comment]=comment.lower()
    return data.map(lower_case_map)

def replace_angle_brackets(data):
    # Remove words in '< >' brackets
    # Ex/ <code> 
    bracket_map = {}
    for comment in data:
        bracket_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            if word.startswith('<') and word.endswith('>'):
                bracket_map[comment]=bracket_map[comment].replace(word,'')
    return data.map(bracket_map)

def expand_contractions(data):
    #'not' contractions without apostrophes 
    contractions={}
    contractions['arent']='are not'
    contractions['cannot']='can not'
    contractions['cant']='can not'
    contractions['couldnt']='could not'
    contractions['didnt']='did not'
    contractions['doesnt']='does not'
    contractions['dont']='do not'
    contractions['hasnt']='has not'
    contractions['havent']='have not'
    contractions['isnt']='is not'
    contractions['shouldnt']='should not'
    contractions['werent']='were not'
    contractions['wont']='will not'
    contractions['wouldnt']='would not'

    # Other cases
    contractions["can't"]='can not'
    contractions["won't"]='will not'
    contractions["where'd"]='where did'
    contractions["i'm"]='i am'
    contractions["it's"]='it is'
    contractions["there's"]='there is'
    contractions["theres"]='there is'
    contractions["how'd"]='how did'
    
    contraction_map = {}
    for comment in data:
        contraction_map[comment]=comment
        for contraction in contractions:
            if contraction in contraction_map[comment]:
                contraction_map[comment]=contraction_map[comment].replace(contraction,contractions[contraction])
        contraction_map[comment]=contraction_map[comment].replace("n't", ' not')
        contraction_map[comment]=contraction_map[comment].replace("'ll", ' will')
        contraction_map[comment]=contraction_map[comment].replace("'d", ' would')
        contraction_map[comment]=contraction_map[comment].replace("'ve", ' have')
    return data.map(contraction_map)

def replace_links(data):
    # Ex/ https://www.veamly.com 
    link_map = {}
    for comment in data:
        link_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            word = word.rstrip('.,!?():<>"')
            link = ''
            if 'https://' in word:
                link = word[word.index('https://'):]
            elif 'http://' in word:
                link = word[word.index('http://'):]
            elif 'www.' in word:
                link = word[word.index('www.'):]
            if link != '':
                link_map[comment]=link_map[comment].replace(link,'')
    return data.map(link_map)

def replace_emails(data):
    # Replace words containing '@' and '.com'
    # Ex/ veamly@gmail.com => <email> 
    email_map = {}
    for comment in data:
        email_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            if '@' in word and '.com' in word:
                email_map[comment]=email_map[comment].replace(word,'')
    return data.map(email_map)

def replace_hashtags(data):
    # Replace words starting with '#'
    # Ex/ #2017 
    hashtag_map = {}
    for comment in data:
        hashtag_map[comment]=comment
        try:
            wordlist = comment.split()
            for word in wordlist:
                if word.startswith('#'):
                    hashtag_map[comment]=hashtag_map[comment].replace(word,'')
        except:
            print(comment)
    return data.map(hashtag_map)

def replace_html_entities(data):
    data = data.str.replace('&lt;','').str.replace('&gt;','')
    data = data.str.replace('&amp;','')
    data = data.str.replace('&quot;','')
    return data

def replace_mentions(data):
    # Replace words starting with '@' without '.com' 
    # Ex/ @brian 
    mention_map = {}
    for comment in data:
        mention_map[comment]=comment
        try:
            wordlist = comment.split()
            for word in wordlist:
                if word.startswith('@'):
                    mention_map[comment]=mention_map[comment].replace(word,'')
        except:
            print(comment)
    return data.map(mention_map)

def replace_punctuation(data):
    import string
    punctuation_map = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    data = data.str.translate(punctuation_map)
    return data

def replace_snippets(data):
    # Replace all text between ''' indicators
    snippet_map = {}
    for comment in data:
        snippet_map[comment]=comment
        while snippet_map[comment].count('```')>1 and snippet_map[comment].count('```')%2==0:
            start = snippet_map[comment].index('```')
            end = start+snippet_map[comment][start+3:].index('```')+6
            snippet = snippet_map[comment][start:end]
            snippet_map[comment]=snippet_map[comment].replace(snippet,'')
    return data.map(snippet_map)

def replace_emojis(data):
    # Ex/ :smile: 
    emoji_map = {}
    for comment in data:
        emoji_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            if word.startswith(':') and word.endswith(':'):
                emoji_map[comment]=emoji_map[comment].replace(word,'')
    return data.map(emoji_map)

def replace_replies(data):
    # Remove all lines starting with '>' 

    reply_map = {}
    # Filter out replies
    for comment in data:
        reply_map[comment]=comment
        # Delete first line if first line is a reply
        if reply_map[comment].startswith('>'):
            if '\n' in reply_map[comment]:
                end = reply_map[comment].index('\n')+2
            else:
                end = len(reply_map[comment])
            reply = reply_map[comment][:end]
            reply_map[comment]=reply_map[comment].replace(reply,'')
        
        # Delete all other lines that are replies
        while '\n>' in reply_map[comment]:
            start = reply_map[comment].index('\n>')
            if '\n' in reply_map[comment][start+3:]:
                end = start+reply_map[comment][start+3:].index('\n')+3
            else:
                end = len(reply_map[comment])
            reply = reply_map[comment][start:end]
            reply_map[comment]=reply_map[comment].replace(reply,'')
    return data.map(reply_map)

def replace_words_containing_numbers(data):
    # Replace words containing a number
    number_map = {}
    for comment in data:
        number_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            for char in word:
                if char.isdigit():
                    number_map[comment]=number_map[comment].replace(word,'')
                    break
    return data.map(number_map)

def trim_whitespace(data):
    trim_map = {}
    for comments in data:
        trim_map[comments]=comments.strip()
    return data.map(trim_map)

def replace_code_lines(data):
    # Label all text between ' indicators as <code>
    # Ex/ 'len(word)' => <code>

    code_map = {}
    for comment in data:
        # Label by ` indicators
        code_map[comment]=comment
        while code_map[comment].count('`')>1 and code_map[comment].count('`')%2==0:
            start = code_map[comment].index('`')
            end = start+code_map[comment][start+1:].index('`')+2
            code = code_map[comment][start:end]
            code_map[comment]=code_map[comment].replace(code,'<code>')
    return data.map(code_map)

def replace_code_misc(data):
    # Miscellaneous methods to label text as <code>
    code_map = {}
    for comment in data:
        code_map[comment]=comment
        wordlist = comment.split()
        for word in wordlist:
            # Label words with length > 16 as <code>
            if len(word) > 16:
                code_map[comment]=code_map[comment].replace(word,'')
            # Label words with combination of letters and numbers as <code>
            elif re.search('[a-zA-Z]', word) and re.search('\d', word):
                code_map[comment]=code_map[comment].replace(word,'<code>')
            # Label words with a period in the middle as <code>
            elif '.' in word.rstrip('."') and '..' not in word:
                if not re.search('e.g', word.lower()) and not re.search('i.e', word.lower()):
                    code_map[comment]=code_map[comment].replace(word,'<code>')
            # Label words with lowercase first letter and uppercase somewhere after first letter as <code>
            else:
                word = word.lstrip('""_:.<{/#')
                word = word.lstrip("'")
                if len(word) > 2 and re.search('[A-Z]', word[2:]) and re.search('[a-z]', word[1]):
                    code_map[comment]=code_map[comment].replace(word,'<code>')
    return data.map(code_map)