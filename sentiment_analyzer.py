import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, opinion_lexicon
import re

# DOWNLOAD THE NECESSARY RESOURCES!
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('stopwords')

class TextAnalyzer:
    def __init__(self):
        pass

    def extract_article_info(self, url):

        """Extracts article title and text from a URL"""

        try:
            response = requests.get(url)  # Sending an HTTP GET request to the URL.
            soup = BeautifulSoup(response.text, 'html.parser')  # Parsing the HTML content of the response.

            # Extracting article title.
            article_title = None
            entry_title = soup.find('h1', class_='entry-title')  # Finding the HTML element with class 'entry-title'.
            if entry_title:
                article_title = entry_title.get_text().strip()  # Extracting and stripping the text of the element.

            # Extracting article text
            article_text = None
            article_content = soup.find('div', class_='td-post-content tagdiv-type')  # Finding the HTML element with
                                                                                # class 'td-post-content tagdiv-type'.
            if article_content:
                article_text = article_content.get_text().strip()  # Extracting and stripping the text of the element.

            return article_title, article_text  # Returns the article title and text(TO BE USED DURING SENTIMENT ANALYSIS)

        except Exception as e:
            print(f"Error occurred while extracting article info from {url}: {e}")   # CATCHING THE EXCEPTION!!!
            return None, None

    def clean_text(self, text):

        """Performs cleaning of the text"""

        tokens = word_tokenize(text)  # Tokenizing the text into words...
        words = [word.lower() for word in tokens if word.isalnum()]  # Removing punctuations and converting to lowercase.
        stop_words = set(stopwords.words('english')) # Getting English stopwords: Here instead of a set, a list can also
        # be used. But using a set offers faster lookup times compared to using a list, especially for large
        # collections of words.

        words = [word for word in words if word not in stop_words]  # Removing the stopwords!
        cleaned_text = ' '.join(words)  # Joining the words into a single string...

        return cleaned_text  # Returns the cleaned text.

    def sentiment_analysis(self, text):

        """Performs sentiment analysis"""

        positive_words = set(opinion_lexicon.words("positive-words.txt"))  # Getting positive words from the opinion lexicon.
        negative_words = set(opinion_lexicon.words("negative-words.txt"))  # Getting negative words from the opinion lexicon.
        positive_score = sum(1 for word in text if word in positive_words)   # Counting positive words in the text.

        negative_score = sum(1 for word in text if word in negative_words)  # Counting negative words in the text.

        # Calculates polarity score.
        polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
        # Calculates subjectivity score.
        subjectivity_score = (positive_score + negative_score) / (len(text) + 0.000001)

        return positive_score, negative_score, polarity_score, subjectivity_score  # Returns sentiment analysis scores!

    def readability_analysis(self, text):

        """Performs readability analysis"""

        sentences = sent_tokenize(text)  # Tokenizing the text into sentence
        num_sentences = len(sentences)  # Counting the number of sentences.
        total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)  # Counting the total number of words.

        avg_words_per_sentence = total_words / num_sentences  # Calculates average words per sentence.
        words = self.clean_text(text).split()  # Cleaning the text and splitting into words.

        # Counting complex words based on length and syllable count.
        complex_word_count = sum(1 for word in words if len(word) > 6 and self.count_syllables(word) > 3)
        # Calculates the fog index.
        fog_index = 0.4 * (avg_words_per_sentence + (complex_word_count / len(words)))

        return avg_words_per_sentence, complex_word_count, fog_index  # Returns readability analysis metrics.

    def count_syllables(self, word):

        """Counts syllables in a word"""

        vowels = 'aeiou'
        syllables = 0
        prev_char_was_vowel = False
        for char in word:
            if char in vowels and not prev_char_was_vowel:
                syllables += 1
                prev_char_was_vowel = True
            elif char not in vowels:
                prev_char_was_vowel = False

        return syllables  # Returning the syllable count.

    def syllable_count_per_word(self, text):

        """Calculates syllable count per word"""

        words = self.clean_text(text).split()  # Cleaning the text and splitting into words...
        total_syllables = sum(self.count_syllables(word) for word in words)   # Counting total syllables.

        return total_syllables / len(words)  # Calculates average syllable count per word.

    def count_personal_pronouns(self, text):

        """Counts personal pronouns"""

        pronouns = ['i', 'we', 'my', 'ours', 'us']
        regex_pattern = r'\b(?:{})\b'.format('|'.join(pronouns))  # Creating a regex pattern for matching pronouns.
        personal_pronouns = re.findall(regex_pattern, text.lower())  # Finds personal pronouns in the text.

        return len(personal_pronouns)  # Returns the count of personal pronouns.

    def average_word_length(self, text):

        """Calculates average word length"""

        words = self.clean_text(text).split()
        total_characters = sum(len(word) for word in words)

        return total_characters / len(words)  # Calculates average word length...

    def analyze_text(self, text):

        """Analyzes text and returns various metrics"""

        cleaned_text = self.clean_text(text)
        tokenized_text = word_tokenize(cleaned_text)
        positive_score, negative_score, polarity_score, subjectivity_score = self.sentiment_analysis(tokenized_text)
        avg_words_per_sentence, complex_word_count, fog_index = self.readability_analysis(cleaned_text)
        syllable_per_word = self.syllable_count_per_word(cleaned_text)
        personal_pronouns = self.count_personal_pronouns(cleaned_text)
        avg_word_length = self.average_word_length(cleaned_text)

        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_words_per_sentence,
            'PERCENTAGE OF COMPLEX WORDS': complex_word_count / len(tokenized_text),
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': len(tokenized_text),
            'SYLLABLE PER WORD': syllable_per_word,
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': avg_word_length
        }


analyzer = TextAnalyzer() # Instantiate TextAnalyzer


def main():
    df = pd.read_excel('input.xlsx')

    results = []

    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        article_title, article_text = analyzer.extract_article_info(url)

        if article_title and article_text:
            analysis_results = analyzer.analyze_text(article_text)
            analysis_results['URL_ID'] = url_id
            analysis_results['URL'] = url
            results.append(analysis_results)

    output_df = pd.DataFrame(results)
    output_df.to_excel('Output.xlsx', index=False)
    print("Extraction completed. Output saved to Output.xlsx")


if __name__ == "__main__":
    main()
