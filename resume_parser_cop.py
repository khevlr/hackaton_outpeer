import fitz 
import openai
from bs4 import BeautifulSoup
import json
from prompts import RESUME_SKILLS_PROMPT, VACANCY_ANALYSIS_PROMPT, CV_ANALYSIS_PROMPT, TEST_GENERATION_PROMPT, NEW_RESUME_PROMPT
from dotenv import load_dotenv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import time 


load_dotenv()

CHAT_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = CHAT_API_KEY

tokenizer = AutoTokenizer.from_pretrained('ai-forever/ru-en-ROSBERTa')
model = AutoModel.from_pretrained('ai-forever/ru-en-ROSBERTa')

def analyze_resume(resume_file):
    text = ""
    doc = fitz.open(resume_file)
    text = "\n".join([page.get_text() for page in doc])
    prompt = CV_ANALYSIS_PROMPT.format(text=text)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": "Ты эксперт в аналитике резюме и вакансий."},
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

def extract_resume_skills(resume_file, vacancy_name):
    with open('vacancies/' + vacancy_name, 'r') as f:
        vacancy_text = json.load(f)
    vacancy_description = f"""
        Описание вакансии:
        {vacancy_text['summary']}
        Требования для вакансии:
        {vacancy_text['skills']}
    """
    text = ""
    doc = fitz.open(resume_file)
    text = "\n".join([page.get_text() for page in doc])

    prompt = RESUME_SKILLS_PROMPT.format(text=text, vacancy_text=vacancy_description)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": "Ты эксперт в аналитике резюме и вакансий."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
        

def find_main_text_block(soup):
    candidates = soup.find_all(['article', 'main', 'section', 'div'])
    best_block = ""
    max_score = 0

    for tag in candidates:
        text = tag.get_text(separator="\n", strip=True)
        score = len(text)
        if score > max_score and score > 100: 
            max_score = score
            best_block = text

    return best_block if best_block else soup.get_text(separator="\n", strip=True)


def get_page_content(url):
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

        return find_main_text_block(soup)
    finally:
        driver.quit()


def analyze_vacancy(vacancy_text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": "Ты эксперт в аналитике резюме и вакансий."},
            {"role": "user", "content": VACANCY_ANALYSIS_PROMPT.format(vacancy_text=vacancy_text)}
        ]
    )
    return json.loads(response.choices[0].message.content)


def roberta_similarity(text):
    tokenized_data = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v for k, v in tokenized_data.items()}
    with torch.no_grad():
        output = model(**inputs)
    embeddings = output.last_hidden_state
    mask = inputs['attention_mask']
    expanded_mask = mask.unsqueeze(-1).expand_as(embeddings).float()
    masked_embeddings = embeddings * expanded_mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = torch.sum(mask, dim=1).unsqueeze(-1)
    sum_mask = torch.clamp(sum_mask, min=0.0)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings.squeeze(0)

def calculate_semantic_score(embedding1, embedding2):
    embedding1_np = embedding1.numpy().reshape(1, -1)
    embedding2_np = embedding2.numpy().reshape(1, -1)
    similarity_score = cosine_similarity(embedding1_np, embedding2_np)
    return similarity_score

def skills_similarity(resume_skills, vacancy_skills):
    vacancy_skills_lower = vacancy_skills.lower()
    counter = 0
    for skill in resume_skills:
        if skill.lower() in vacancy_skills_lower:
            counter += 1
    return counter / len(resume_skills)

def find_vacancy_for_resume(resume_json):
    similarity_scores = []
    for file in os.listdir('vacancies'):
        with open('vacancies/' + file, 'r') as f:
            vacancy_data = json.load(f)
        semantic_score = calculate_semantic_score(roberta_similarity(vacancy_data['summary']), roberta_similarity(resume_json['summary']))
        skills_score = skills_similarity(resume_json['skills'], vacancy_data['skills'])
        total_score = semantic_score * 0.8 + skills_score * 0.2
        similarity_scores.append({
            'vacancy_name': file,
            'semantic_score': semantic_score,
            'skills_score': skills_score,
            'total_score': total_score
        })
    return sorted(similarity_scores, key=lambda x: x['total_score'], reverse=True)[:5]

def test_generator(vacancy_name):
    with open('vacancies/' + vacancy_name, 'r') as f:
        vacancy_text = json.load(f)
    prompt = TEST_GENERATION_PROMPT.format(skills_table=vacancy_text['skills'])

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты эксперт в создании тестов."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    
    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print("Ошибка парсинга JSON:", e)
        print("Содержимое:", raw_output)
        return None

    return parsed_output

def check_answers(human_answers, real_answers, questions):
    final_score = 0
    list_of_questions = []

    for idx, (human_answer, real_answer) in enumerate(zip(human_answers, real_answers)):
        question_score = 0

        if len(real_answer) == 1:
            if human_answer and human_answer[0] == real_answer[0]:
                question_score = 1

        else:
            for option in human_answer:
                if option in real_answer:
                    question_score += 0.5

        final_score += question_score

        list_of_questions.append({
            "question": questions[idx],
            "correct_answer": real_answer,
            "user_answer": human_answer,
            "score": question_score
        })

    return final_score, list_of_questions

def new_resume(resume, vacancy):
    with open('vacancies/' + vacancy, 'r') as f:
        vacancy_text = json.load(f)

    prompt = NEW_RESUME_PROMPT.format(resume_text=resume, vacancy_description=vacancy_text['summary'])

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты HR-специалист с опытом составления резюме под конкретные вакансии."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    return content

