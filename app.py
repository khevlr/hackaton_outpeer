import streamlit as st
import os
from resume_parser_cop import (
    analyze_resume,
    find_vacancy_for_resume,
    extract_resume_skills,
    test_generator,
    check_answers,
    new_resume
)

st.set_page_config(page_title="Resume Analyzer", layout="wide")

def initialize_session_state():
    if "resume_analyzed" not in st.session_state:
        st.session_state.resume_analyzed = False
    if "resume_analysis" not in st.session_state:
        st.session_state.resume_analysis = None
    if "matching_vacancies" not in st.session_state:
        st.session_state.matching_vacancies = None
    if "selected_vacancy" not in st.session_state:
        st.session_state.selected_vacancy = None
    if "vacancy_display_name" not in st.session_state:
        st.session_state.vacancy_display_name = None
    if "skills_analysis" not in st.session_state:
        st.session_state.skills_analysis = None
    if "test_questions" not in st.session_state:
        st.session_state.test_questions = None
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
    if "test_results" not in st.session_state:
        st.session_state.test_results = None
    if "resume_path" not in st.session_state:
        st.session_state.resume_path = None
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "improved_resume" not in st.session_state:
        st.session_state.improved_resume = None
    if "vacancy_name_to_file" not in st.session_state:
        st.session_state.vacancy_name_to_file = {}

def analyze_resume_callback():
    doc = fitz.open(st.session_state.resume_path)
    st.session_state.resume_text = "\n".join([page.get_text() for page in doc])
    
    with st.spinner("Анализ резюме..."):
        st.session_state.resume_analysis = analyze_resume(st.session_state.resume_path)
        st.session_state.resume_analyzed = True
    
    with st.spinner("Поиск подходящих вакансий..."):
        st.session_state.matching_vacancies = find_vacancy_for_resume(st.session_state.resume_analysis)
        
        st.session_state.vacancy_name_to_file = {}
        for vacancy in st.session_state.matching_vacancies:
            file_name = vacancy["vacancy_name"]
            
            try:
                parts = file_name.replace('.json', '').split('_')
                if len(parts) >= 2:
                    display_name = f"{parts[0]} - {' '.join(parts[1:])}"
                else:
                    display_name = file_name
            except:
                display_name = file_name
                
            st.session_state.vacancy_name_to_file[display_name] = file_name

def select_vacancy_callback():
    selected_display = st.session_state.vacancy_selector
    
    if selected_display in st.session_state.vacancy_name_to_file:
        st.session_state.selected_vacancy = st.session_state.vacancy_name_to_file[selected_display]
        st.session_state.vacancy_display_name = selected_display
    
    with st.spinner("Анализ резюме и вакансии..."):
        st.session_state.skills_analysis = extract_resume_skills(
            st.session_state.resume_path, 
            st.session_state.selected_vacancy
        )
    
    with st.spinner("Генерация теста..."):
        test_data = test_generator(st.session_state.selected_vacancy)
        
        if test_data and isinstance(test_data, list):
            valid_questions = []
            for q in test_data:
                if isinstance(q, dict) and 'question' in q and 'options' in q and 'answer' in q:
                    if isinstance(q['options'], list) and len(q['options']) > 0:
                        valid_questions.append(q)
            
            st.session_state.test_questions = valid_questions
            st.session_state.user_answers = [[] for _ in range(len(valid_questions))]
            
            for i in range(len(valid_questions)):
                if f"q{i}" not in st.session_state:
                    st.session_state[f"q{i}"] = None
        else:
            st.session_state.test_questions = []
            st.session_state.user_answers = []
            st.error("Failed to generate valid test questions. Please try another vacancy.")
    
    st.session_state.improved_resume = None

def generate_improved_resume():
    with st.spinner("Генерация нового резюме..."):
        improved = new_resume(st.session_state.resume_text, st.session_state.selected_vacancy)
        st.session_state.improved_resume = improved

def update_answer(i, multiple=False):
    current_value = st.session_state.get(f"q{i}", None)
    
    if multiple:
        st.session_state.user_answers[i] = current_value if current_value else []
    else:
        st.session_state.user_answers[i] = [current_value] if current_value else []

def submit_test():
    if not st.session_state.test_questions or len(st.session_state.test_questions) == 0:
        st.error("No valid test questions to check")
        return
        
    true_answers = [question['answer'] for question in st.session_state.test_questions]
    questions_text = [question['question'] for question in st.session_state.test_questions]
    
    final_score, question_results = check_answers(
        st.session_state.user_answers,
        true_answers,
        questions_text
    )
    
    st.session_state.test_results = {
        "final_score": final_score,
        "question_results": question_results,
        "max_score": len(st.session_state.test_questions)
    }

def main():
    st.title("Анализатор резюме")
    
    initialize_session_state()
    
    uploaded_file = st.file_uploader("Загрузите свое резюме (PDF)", type=["pdf"])
    
    if uploaded_file:
        if st.session_state.resume_path is None:
            os.makedirs('uploaded_files', exist_ok=True)
            file_path = os.path.join('uploaded_files', f"{uploaded_file.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.resume_path = file_path
        
        if not st.session_state.resume_analyzed:
            if st.button("Анализировать"):
                analyze_resume_callback()
        
        if st.session_state.resume_analyzed:
            st.subheader("Анализ резюме")
            st.write("Summary:")
            st.write(st.session_state.resume_analysis["summary"])
            
            st.write("Skills:")
            st.write(", ".join(st.session_state.resume_analysis["skills"]))
            
            st.subheader("Топ подходящих вакансий")
            
            vacancy_display_options = list(st.session_state.vacancy_name_to_file.keys())
            
            st.selectbox(
                "Выберите вакансию для детального анализа:",
                vacancy_display_options,
                key="vacancy_selector",
                on_change=select_vacancy_callback
            )
            
            if st.session_state.selected_vacancy:
                st.subheader(f"Анализ навыков для {st.session_state.vacancy_display_name}")
                st.markdown(st.session_state.skills_analysis)
                
                if st.session_state.test_questions and len(st.session_state.test_questions) > 0:
                    st.subheader("Тест")
                    st.write("Выберите ответ:")
                    
                    for i, question in enumerate(st.session_state.test_questions):
                        st.markdown(f"**Q{i+1}: {question['question']}**")
                        
                        options = question.get('options', [])
                        if not isinstance(options, list) or len(options) == 0:
                            st.error(f"Question {i+1} has invalid options")
                            continue
                            
                        multiple = isinstance(question.get('answer', []), list) and len(question.get('answer', [])) > 1
                        
                        try:
                            if multiple:
                                st.multiselect(
                                    f"Выберите ответ {i+1}",
                                    options,
                                    key=f"q{i}",
                                    on_change=update_answer,
                                    args=(i, True)
                                )
                            else:
                                st.radio(
                                    f"Выберите ответ {i+1}",
                                    options,
                                    key=f"q{i}",
                                    on_change=update_answer,
                                    args=(i, False)
                                )
                        except Exception as e:
                            st.error(f"Error displaying question {i+1}: {str(e)}")
                    
                    if st.button("Завершить тест", on_click=submit_test):
                        pass  
                    
                    if st.session_state.test_results:
                        results = st.session_state.test_results
                        percentage = round((results["final_score"] / results["max_score"]) * 100, 2)
                        
                        st.subheader("Результаты")
                        st.markdown(f"**Ваш результат: {results['final_score']}/{results['max_score']} ({percentage}%)**")
                        
                        for i, result in enumerate(results["question_results"]):
                            with st.expander(f"Вопрос {i+1}: {result['score']} points"):
                                st.write(f"**Вопрос:** {result['question']}")
                                user_ans = result.get('user_answer', [])
                                correct_ans = result.get('correct_answer', [])
                                st.write(f"**Ваш ответ:** {', '.join(user_ans) if isinstance(user_ans, list) else str(user_ans)}")
                                st.write(f"**Правильный ответ:** {', '.join(correct_ans) if isinstance(correct_ans, list) else str(correct_ans)}")
                        
                        st.subheader("Улучшение резюме")
                        st.write("Получите улучшенную версию вашего резюме, адаптированную под эту вакансию:")
                        
                        if st.button("Создайте улучшенное резюме"):
                            generate_improved_resume()
                        
                        if st.session_state.improved_resume:
                            st.markdown("### Ваше улучшенное резюме")
                            st.markdown(st.session_state.improved_resume)
                            
                            improved_resume_txt = st.session_state.improved_resume
                            st.download_button(
                                label="Скачать улучшенное резюме",
                                data=improved_resume_txt,
                                file_name="improved_resume.md",
                                mime="text/markdown"
                            )
                else:
                    st.warning("No test questions were generated for this vacancy. Please try another vacancy.")

if __name__ == "__main__":
    import fitz  
    main() 