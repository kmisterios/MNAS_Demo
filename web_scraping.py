import urllib.request, sys, re
import xmltodict, json

import os
from googlesearch import search
from contextlib import contextmanager
import re
import signal
import requests
from bs4 import BeautifulSoup
from scipy.spatial import distance
from settings.diffbot import client
# from settings.scraper import *
# from settings.translator import translator
from deep_translator import GoogleTranslator
import pandas as pd
import pickle
from tqdm import tqdm

TIMEOUT = 5

TARGET_LANGS = ['en', 'fr', 'de', 'es', 'ru']



FORBIDDEN_TYPES = ['pdf', 'txt', '.gz', 'rar', 'doc', 'xml']
NUM_RESULT_PAGES = 2


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        

def get_alexa_rank(link):
    try:
        xml = urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url={}'.format(link)).read()
        result= xmltodict.parse(xml)
         
        data = json.dumps(result).replace("@","")
        data_tojson = json.loads(data)
        url = data_tojson["ALEXA"]["SD"][1]["POPULARITY"]["URL"]
        rank = int(data_tojson["ALEXA"]["SD"][1]["POPULARITY"]["TEXT"])
    except:
        print('Link issue.')
        rank = sys.maxsize
    
    return rank


def generate_request_simple(news):
    return news['headline']


def generate_request_cleaned(news):
    return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", news['headline'])




def extract_search_results(request, lang, num_news):
    # search_results = google.search(request, NUM_RESULT_PAGES)
    
    result_news = []
    
    try:
        search_results = search(request, lang=lang, num=num_news, stop=num_news, pause=1,
         user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36")
        for search_result in search_results:
            search_news_headline = ""
            search_news_content = ""

            search_link = search_result  # .link
            print(search_result)
            if search_link is None:
                continue
            if search_link[-3:] in FORBIDDEN_TYPES:
                continue

            try:
                try:
                    # with time_limit(TIMEOUT):
                    response = requests.get(search_link)
                except TimeoutException as e:
                    print("Timed out!")
                    continue

                # scraping content
                search_news_content = client.article(search_link)['objects'][0]['text']
                # scraping headline
                soup = BeautifulSoup(response.text, features= "html.parser")
                res_meta = soup.find_all('meta', property="og:title")
                res_title = soup.find_all('title')
                if not (len(res_meta) > 0 or len(res_title) > 0):
                    continue
                if len(res_meta) > 0:
                    search_news_headline = res_meta[0].attrs['content']
                elif len(res_title) > 0:
                    search_news_headline = res_title[0].contents[0]

                search_link_rank = get_alexa_rank(search_link)
                ############################

                result_news.append({'url': search_link,
                                    'headline': search_news_headline.strip(),
                                    'content': search_news_content.strip(),
                                    'alexa_rank': search_link_rank})

            except Exception as e:
                print(e)

    except Exception as e:
        print(e)
                
    return result_news



def manual_news_scraping(save_fake_to, query, num_news):
    query_list = [query]

    scraped_results = {}
    for index, news in enumerate(query_list):
        scraped_results[index] = {}

        lang_list = ["en","fr", "de", "es", "ru"]
        for lang in lang_list:
            scraped_results[index][lang] = []
            translator = GoogleTranslator(source='auto', target=lang)
            try:
                translated_news = {'headline': translator.translate(news)}
            except:
                print('Translation issue.')
                translated_news = {'headline': news}

            request = generate_request_cleaned(translated_news)
            search_results = extract_search_results(request, lang, num_news)

            for search_result in search_results:
                scraped_results[index][lang].append(search_result)

            # saving scraped multilingual evidence
            with open(save_fake_to + '.pkl', 'wb') as file:
                pickle.dump(scraped_results, file, pickle.HIGHEST_PROTOCOL)
                
                