
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time 


def get_vacancies_links(url):

    print(url)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)
        
        page_content = driver.page_source
        soup = BeautifulSoup(page_content, "html.parser")

        links = soup.find_all('a', href=True)
    
        return [link['href'] for link in links]
    
    finally:
        driver.quit()

def create_search_link(job_title, page=0):
    
    job_title_split = job_title.split(' ')
    job_title_add = '+'.join(job_title_split)
    
    return f"https://almaty.hh.kz/search/vacancy?text={job_title_add}&area=40&page={page}"

job_titles = ["Data Analyst", "Data Scientist", "Data Engineer", "Аналитик данных", "Python Engineer", "Python Developer", "Python разработчик"]

# all_links = []
# for title in job_titles:
#     title_links = []
#     page_num = 0
#     while page_num < 2: 
#         links = get_vacancies_links(create_search_link(title, page_num))
#         print(links)
#         vacancy_links = [link for link in links if link.startswith('https://almaty.hh.kz/vacancy/')]
#         if not vacancy_links:
#             break
#         page_num += 1
#         title_links.extend(vacancy_links)

#     with open(f'{title}_vacancy_links.json', 'w') as f:
#         json.dump(title_links, f)

#     all_links.extend(title_links)

# read titles files and take the first 40 links

all_links = []

for title in job_titles:

    with open(f'{title}_vacancy_links.json', 'r') as f:
        title_links = json.load(f)

    all_links.extend(title_links[:40])

#example link https://almaty.hh.kz/vacancy/120047204?query=Data+Analyst&hhtmFrom=vacancy_search_list
#remove duplicates by id

all_links_unique = []
all_links_ids = set()

for link in all_links:

    job_id = link.split('/')[-1]
    job_id = job_id.split('?')[0]

    print(job_id)

    if job_id not in all_links_ids:
        all_links_ids.add(job_id)
        all_links_unique.append(link)

with open('all_vacancy_links_short.json', 'w') as f:
    json.dump(all_links_unique, f)





