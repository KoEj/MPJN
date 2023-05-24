import requests
import random
import json
from bs4 import BeautifulSoup
import time


# Zadanie 1
# url = "https://wolnelektury.pl/api/books/"
# response = requests.get(url).json()
#
# with open('scrapped_books.json', 'w', encoding='utf-8') as f:
#     json.dump(response, f, indent=2)
#
# random_books = random.sample(response, 20)
#
# for book in random_books:
#     print(f"Title: {book['title']}")
#     print(f"Autor: {book['author']}")
#     print(f"Genre: {book['genre']}")
#     print(f"Epoch: {book['epoch']}")
#     href_response = requests.get(book['href']).json()
#
#     if href_response['txt'] != '':
#         txt_response = requests.get(href_response['txt'])
#         print(f"Content: {txt_response.text}")
#     else:
#         print(f"Missing URL to txt, PDF here {href_response['pdf']}")
#
#     print("\n")

# Innym przykladem otwartego API moze byc api od mbanku
# https://developer.api.mbank.pl/
# np. sprawdzanie salda zapytaniem czy realizacja platnosci przez dostawce
# uslug z twojego rachunku (np. przy platnosci przez internet wykorzystanie opcji mbank)

# Zadanie 2
url = "https://gazetawroclawska.pl/wiadomosci/"
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')
# title = soup.find(class_='atomsListingArticleTileWithSeparatedLink')
title = soup.find('h2')
title = title.find('b')

# print(title.text)

# Plik robots.txt zawiera uprawnienia i zdefiniowane dostępy do danej podstrony (lokalizacji) przez boty (np. przez scrawling)
# Komendy użyte w tym pliku to (łącznie z krótkim objaśnieniem)
# user-agent oznacza nazwe robota
# disallow - zapobiega mozliwosci skrobania tresci w konkretnych plikach/podstronach
# allow - przeciwienstwo do metody wyzej; ew. nadpisuje disallow
# sitemap - pokazuje lokalizacje mapy witryny

# Zadanie 3
for link in soup.find_all('a'):
    href = link.get('href')

    if href and href.startswith('/wiadomosci'):
        href = "https://gazetawroclawska.pl" + href

        article_response = requests.get(href)
        article_soup = BeautifulSoup(article_response.text, 'html.parser')

        for article in article_soup.find_all('a'):
            # print(article['href'])

            article_href = article.get('href')
            if "https://gazetawroclawska.pl" and "https://" not in article_href:
                article_href = "https://gazetawroclawska.pl" + article_href
                print(article_href)

                article_response = requests.get(article_href)
                article_soup = BeautifulSoup(article_response.text, 'html.parser')

                article_title = article_soup.find('h1')
                article_txt = article_soup.find(class_='atomsArticleLead')
                print(article_title)
                print(article_txt)

                time.sleep(2)

# Opóźnienie powinno się ustawić, aby nie było sytuacji kiedy w trakcie bardzo krótkiego czasu
# wyśle się bardzo duża ilość zapytań - wówczas strona może nałożyć blokadę na użytkownika
# albo timeouta na requesty albo uznać to za jawny atak i zbanować użytkownika na adres IP
