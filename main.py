import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sentiment_analyzer import analyzer  # Import the analyzer from sentiment_analyzer module...


class ArticleScraper:
    def __init__(self, input_file):
        self.df = pd.read_excel(input_file)  # The input file here is an Excel file...

    def extract_article_info(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            article_title = None
            entry_title = soup.find('h1', class_='entry-title')
            if entry_title:
                article_title = entry_title.get_text().strip()

            article_text = None
            article_content = soup.find('div', class_='td-post-content tagdiv-type')
            if article_content:
                article_text = article_content.get_text().strip()

            return article_title, article_text

        except Exception as e:
            print(f"Error occurred while extracting article info from {url}: {e}")
            return None, None

    def scrape_articles(self):
        articles = []
        for index, row in self.df.iterrows():
            url = row['URL']
            article_title, article_text = self.extract_article_info(url)
            if article_title and article_text:
                cleaned_text = analyzer.clean_text(article_text)
                tokenized_text = word_tokenize(cleaned_text)
                positive_score, negative_score, polarity_score, subjectivity_score = analyzer.sentiment_analysis(tokenized_text)
                avg_words_per_sentence, complex_word_count, fog_index = analyzer.readability_analysis(cleaned_text)
                syllable_per_word = analyzer.syllable_count_per_word(cleaned_text)
                personal_pronouns = analyzer.count_personal_pronouns(cleaned_text)
                avg_word_length = analyzer.average_word_length(cleaned_text)

                articles.append({
                    'URL': url,
                    'Article_Title': article_title,
                    'Article_Text': article_text,
                    'Positive_Score': positive_score,
                    'Negative_Score': negative_score,
                    'Polarity_Score': polarity_score,
                    'Subjectivity_Score': subjectivity_score,
                    'Avg_Words_Per_Sentence': avg_words_per_sentence,
                    'Complex_Word_Count': complex_word_count,
                    'Fog_Index': fog_index,
                    'Syllable_Per_Word': syllable_per_word,
                    'Personal_Pronouns': personal_pronouns,
                    'Avg_Word_Length': avg_word_length
                })

        output_df = pd.DataFrame(articles)
        output_df.to_excel('extracted_articles.xlsx', index=False)

        print("Extraction completed. Output saved to extracted_articles.xlsx")


def main():
    scraper = ArticleScraper('input.xlsx')
    scraper.scrape_articles()


if __name__ == "__main__":
    main()
