import streamlit as st
import os
import fitz
from classes import ResumeParser, SemanticAnalyzer, TestGenerator, ResumeOptimizer
from vector_store import build_vector_db, search_similar_vacancies

# Russian translations dictionary
translations = {
    "Resume Analyzer": "Анализатор Резюме",
    "Initialize Vector Database": "Инициализировать Базу Данных Векторов",
    "Refresh Vector Database": "Обновить Базу Данных Векторов",
    "Vector database initialized successfully!": "База данных векторов успешно инициализирована!",
    "Vector database refreshed successfully!": "База данных векторов успешно обновлена!",
    "Upload your resume (PDF)": "Загрузите ваше резюме (PDF)",
    "Analyze Resume": "Анализировать Резюме",
    "Resume Analysis": "Анализ Резюме",
    "Summary:": "Резюме:",
    "Skills:": "Навыки:",
    "Top Matching Vacancies": "Подходящие Вакансии",
    "Select a vacancy to see detailed analysis:": "Выберите вакансию для детального анализа:",
    "Choose a vacancy": "Выберите вакансию",
    "View Original Vacancy Posting": "Просмотреть Оригинал Вакансии",
    "Skills Analysis for": "Анализ Навыков для",
    "Technical Test": "Технический Тест",
    "Select the correct answer(s) for each question:": "Выберите правильный ответ(ы) для каждого вопроса:",
    "Select answer(s) for question": "Выберите ответ(ы) для вопроса",
    "Select answer for question": "Выберите ответ для вопроса",
    "Submit Test": "Отправить Тест",
    "Test Results": "Результаты Теста",
    "Your score:": "Ваш результат:",
    "Question:": "Вопрос:",
    "Your answer:": "Ваш ответ:",
    "Correct answer:": "Правильный ответ:",
    "Resume Improvement": "Улучшение Резюме",
    "Get an improved version of your resume tailored to this vacancy:": "Получите улучшенную версию вашего резюме, адаптированную к этой вакансии:",
    "Generate Improved Resume": "Сгенерировать Улучшенное Резюме",
    "Your Improved Resume": "Ваше Улучшенное Резюме",
    "Download Improved Resume": "Скачать Улучшенное Резюме",
    "No valid test questions to check": "Нет действительных тестовых вопросов для проверки",
    "Question": "Вопрос",
    "points": "баллов",
    "No test questions were generated for this vacancy. Please try another vacancy.": "Для этой вакансии не были сгенерированы тестовые вопросы. Пожалуйста, попробуйте другую вакансию.",
    "Initializing vector database...": "Инициализация базы данных векторов...",
    "Refreshing vector database...": "Обновление базы данных векторов...",
    "Analyzing your resume...": "Анализ вашего резюме...",
    "Finding matching vacancies...": "Поиск подходящих вакансий...",
    "Analyzing resume against selected vacancy...": "Анализ резюме для выбранной вакансии...",
    "Generating test...": "Генерация теста...",
    "Failed to generate valid test questions. Please try another vacancy.": "Не удалось сгенерировать действительные тестовые вопросы. Пожалуйста, попробуйте другую вакансию.",
    "Generating improved resume for this vacancy...": "Генерация улучшенного резюме для этой вакансии...",
    "Error displaying question": "Ошибка отображения вопроса"
}

def translate(text):
    """Translate text if it exists in the translations dictionary"""
    return translations.get(text, text)

st.set_page_config(page_title=translate("Resume Analyzer"), layout="wide")

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
    if "vacancy_urls" not in st.session_state:
        st.session_state.vacancy_urls = {}
    if "vector_db_initialized" not in st.session_state:
        st.session_state.vector_db_initialized = False

def initialize_vector_db():
    """Initialize the vector database if it doesn't exist."""
    if not st.session_state.vector_db_initialized:
        with st.spinner(translate("Initializing vector database...")):
            build_vector_db()
            st.session_state.vector_db_initialized = True

def refresh_vector_db():
    """Force a rebuild of the vector database to reflect changes in the vacancies folder."""
    with st.spinner(translate("Refreshing vector database...")):
        build_vector_db(force_rebuild=True)
        # Reset any vacancy-related state to ensure fresh data
        st.session_state.selected_vacancy = None
        st.session_state.vacancy_display_name = None
        st.session_state.skills_analysis = None
        st.session_state.test_questions = None
        st.session_state.user_answers = []
        st.session_state.test_results = None
        st.session_state.improved_resume = None
        st.session_state.vacancy_name_to_file = {}
        st.session_state.vacancy_urls = {}
        
        # Re-run analysis if resume was already analyzed
        if st.session_state.resume_analyzed and st.session_state.resume_analysis:
            analyze_resume_callback()

def analyze_resume_callback():
    # Extract the text from the resume
    doc = fitz.open(st.session_state.resume_path)
    st.session_state.resume_text = "\n".join([page.get_text() for page in doc])
    
    with st.spinner(translate("Analyzing your resume...")):
        resume_parser = ResumeParser(st.session_state.resume_path)
        st.session_state.resume_analysis = resume_parser.resume_text
        st.session_state.resume_analyzed = True
    
    # Ensure vector database is initialized
    initialize_vector_db()
    
    with st.spinner(translate("Finding matching vacancies...")):
        # Create embedding for the resume summary
        semantic_analyzer = SemanticAnalyzer()
        resume_embedding = semantic_analyzer.get_embeddings(st.session_state.resume_analysis["summary"])
        
        # Find similar vacancies using vector search
        similar_vacancies = search_similar_vacancies(resume_embedding, n_results=10)
        
        # Format the results for display
        st.session_state.matching_vacancies = []
        st.session_state.vacancy_name_to_file = {}
        st.session_state.vacancy_urls = {}
        
        for vacancy in similar_vacancies:
            filename = vacancy.get('metadata', {}).get('filename', '')
            company = vacancy.get('company_name', '') or vacancy.get('metadata', {}).get('company_name', '')
            position = vacancy.get('position_name', '') or vacancy.get('metadata', {}).get('position_name', '')
            score = vacancy.get('similarity_score', 0)
            vacancy_url = vacancy.get('vacancy_url', '')
            
            # Create a display name
            if company and position:
                display_name = f"{company} - {position}"
            else:
                display_name = filename.replace('.json', '')
            
            st.session_state.vacancy_name_to_file[display_name] = filename
            if vacancy_url:
                st.session_state.vacancy_urls[display_name] = vacancy_url
            
            st.session_state.matching_vacancies.append({
                'vacancy_name': filename,
                'display_name': display_name,
                'total_score': score,
                'vacancy_url': vacancy_url
            })
        
        # Reset selection to ensure the placeholder is shown
        st.session_state.selected_vacancy = None
        st.session_state.vacancy_display_name = None

def select_vacancy_callback():
    # Get the selected display name
    selected_display = st.session_state.vacancy_selector
    
    # If the placeholder was selected, do nothing
    if selected_display == translate("Choose a vacancy"):
        st.session_state.selected_vacancy = None
        st.session_state.vacancy_display_name = None
        return
    
    # Look up the corresponding filename
    if selected_display in st.session_state.vacancy_name_to_file:
        st.session_state.selected_vacancy = st.session_state.vacancy_name_to_file[selected_display]
        st.session_state.vacancy_display_name = selected_display
    
    with st.spinner(translate("Analyzing resume against selected vacancy...")):
        resume_parser = ResumeParser(st.session_state.resume_path)
        st.session_state.skills_analysis = resume_parser.extract_skills_from_resume(st.session_state.selected_vacancy)
    
    with st.spinner(translate("Generating test...")):
        test_generator = TestGenerator(st.session_state.selected_vacancy)
        test_data = test_generator.generate_test()
        
        # Validate test data structure
        if test_data and isinstance(test_data, list):
            # Filter out any invalid questions
            valid_questions = []
            for q in test_data:
                if isinstance(q, dict) and 'question' in q and 'options' in q and 'answer' in q:
                    if isinstance(q['options'], list) and len(q['options']) > 0:
                        valid_questions.append(q)
            
            st.session_state.test_questions = valid_questions
            st.session_state.user_answers = [[] for _ in range(len(valid_questions))]
            
            # Initialize keys for all test questions
            for i in range(len(valid_questions)):
                if f"q{i}" not in st.session_state:
                    st.session_state[f"q{i}"] = None
        else:
            st.session_state.test_questions = []
            st.session_state.user_answers = []
            st.error(translate("Failed to generate valid test questions. Please try another vacancy."))
    
    # Reset improved resume when a new vacancy is selected
    st.session_state.improved_resume = None

def generate_improved_resume():
    with st.spinner(translate("Generating improved resume for this vacancy...")):
        resume_optimizer = ResumeOptimizer(st.session_state.resume_text, st.session_state.selected_vacancy)
        improved = resume_optimizer.optimize_resume()
        st.session_state.improved_resume = improved

def update_answer(i, multiple=False):
    # Get the current value safely
    current_value = st.session_state.get(f"q{i}", None)
    
    if multiple:
        # For multiselect, use the list as is
        st.session_state.user_answers[i] = current_value if current_value else []
    else:
        # For radio, wrap in a list if it exists
        st.session_state.user_answers[i] = [current_value] if current_value else []

def submit_test():
    if not st.session_state.test_questions or len(st.session_state.test_questions) == 0:
        st.error(translate("No valid test questions to check"))
        return
        
    true_answers = [question['answer'] for question in st.session_state.test_questions]
    questions_text = [question['question'] for question in st.session_state.test_questions]
    
    test_generator = TestGenerator(st.session_state.selected_vacancy)
    final_score, question_results = test_generator.check_answers(
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
    st.title(translate("Resume Analyzer"))
    
    initialize_session_state()
    
    # Initialize vector DB on app startup
    if not st.session_state.vector_db_initialized:
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(translate("Initialize Vector Database")):
                initialize_vector_db()
                st.success(translate("Vector database initialized successfully!"))
    else:
        # Add a refresh button if the database is already initialized
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(translate("Refresh Vector Database")):
                refresh_vector_db()
                st.success(translate("Vector database refreshed successfully!"))
    
    # Step 1: Upload resume
    uploaded_file = st.file_uploader(translate("Upload your resume (PDF)"), type=["pdf"])
    
    if uploaded_file:
        # Save the uploaded file to a more permanent location
        if st.session_state.resume_path is None:
            # Create a directory for uploaded files if it doesn't exist
            os.makedirs('uploaded_files', exist_ok=True)
            # Save file with a unique name
            file_path = os.path.join('uploaded_files', f"{uploaded_file.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.resume_path = file_path
        
        if not st.session_state.resume_analyzed:
            if st.button(translate("Analyze Resume")):
                analyze_resume_callback()
        
        if st.session_state.resume_analyzed:
            # Display resume analysis
            st.subheader(translate("Resume Analysis"))
            st.write(translate("Summary:"))
            st.write(st.session_state.resume_analysis["summary"])
            
            st.write(translate("Skills:"))
            st.write(", ".join(st.session_state.resume_analysis["skills"]))
            
            # Display vacancies and let user choose
            st.subheader(translate("Top Matching Vacancies"))
            
            # Create a list of display names and add a placeholder
            vacancy_display_options = [translate("Choose a vacancy")] + list(st.session_state.vacancy_name_to_file.keys())
            
            selected_display = st.selectbox(
                translate("Select a vacancy to see detailed analysis:"),
                vacancy_display_options,
                key="vacancy_selector",
                on_change=select_vacancy_callback,
                index=0  # Default to the placeholder
            )
            
            # Display vacancy link if available as a button
            if selected_display != translate("Choose a vacancy") and selected_display in st.session_state.vacancy_urls:
                vacancy_url = st.session_state.vacancy_urls[selected_display]
                if vacancy_url:
                    # Style the button to make it bigger and more visible
                    st.markdown(
                        f"""
                        <div style="margin: 15px 0;">
                            <a href="{vacancy_url}" target="_blank" style="
                                display: inline-block;
                                padding: 12px 20px;
                                background-color: #4CAF50;
                                color: white;
                                text-align: center;
                                text-decoration: none;
                                font-size: 16px;
                                border-radius: 4px;
                                cursor: pointer;
                                transition: background-color 0.3s;
                                width: 100%;
                            ">{translate("View Original Vacancy Posting")}</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            if st.session_state.selected_vacancy:
                # Display skills analysis
                st.subheader(f"{translate('Skills Analysis for')} {st.session_state.vacancy_display_name}")
                st.markdown(st.session_state.skills_analysis)
                
                if st.session_state.test_questions and len(st.session_state.test_questions) > 0:
                    st.subheader(translate("Technical Test"))
                    st.write(translate("Select the correct answer(s) for each question:"))
                    
                    # Display questions
                    for i, question in enumerate(st.session_state.test_questions):
                        st.markdown(f"**Q{i+1}: {question['question']}**")
                        
                        # Ensure options are available and valid
                        options = question.get('options', [])
                        if not isinstance(options, list) or len(options) == 0:
                            st.error(f"{translate('Question')} {i+1} {translate('has invalid options')}")
                            continue
                            
                        # Determine if multiple answers are allowed
                        multiple = isinstance(question.get('answer', []), list) and len(question.get('answer', [])) > 1
                        
                        try:
                            if multiple:
                                st.multiselect(
                                    f"{translate('Select answer(s) for question')} {i+1}",
                                    options,
                                    key=f"q{i}",
                                    on_change=update_answer,
                                    args=(i, True)
                                )
                            else:
                                st.radio(
                                    f"{translate('Select answer for question')} {i+1}",
                                    options,
                                    key=f"q{i}",
                                    on_change=update_answer,
                                    args=(i, False)
                                )
                        except Exception as e:
                            st.error(f"{translate('Error displaying question')} {i+1}: {str(e)}")
                    
                    # Submit button for test
                    if st.button(translate("Submit Test"), on_click=submit_test):
                        pass  # The actual work is done in the callback
                    
                    # Display results if available
                    if st.session_state.test_results:
                        results = st.session_state.test_results
                        percentage = round((results["final_score"] / results["max_score"]) * 100, 2)
                        
                        st.subheader(translate("Test Results"))
                        st.markdown(f"**{translate('Your score')}: {results['final_score']}/{results['max_score']} ({percentage}%)**")
                        
                        # Display detailed results
                        for i, result in enumerate(results["question_results"]):
                            with st.expander(f"{translate('Question')} {i+1}: {result['score']} {translate('points')}"):
                                st.write(f"**{translate('Question')}:** {result['question']}")
                                user_ans = result.get('user_answer', [])
                                correct_ans = result.get('correct_answer', [])
                                st.write(f"**{translate('Your answer')}:** {', '.join(user_ans) if isinstance(user_ans, list) else str(user_ans)}")
                                st.write(f"**{translate('Correct answer')}:** {', '.join(correct_ans) if isinstance(correct_ans, list) else str(correct_ans)}")
                        
                        # Add resume improvement section
                        st.subheader(translate("Resume Improvement"))
                        st.write(translate("Get an improved version of your resume tailored to this vacancy:"))
                        
                        if st.button(translate("Generate Improved Resume")):
                            generate_improved_resume()
                        
                        if st.session_state.improved_resume:
                            st.markdown(f"### {translate('Your Improved Resume')}")
                            st.markdown(st.session_state.improved_resume)
                            
                            # Option to download the improved resume as a text file
                            improved_resume_txt = st.session_state.improved_resume
                            st.download_button(
                                label=translate("Download Improved Resume"),
                                data=improved_resume_txt,
                                file_name="improved_resume.md",
                                mime="text/markdown"
                            )
                else:
                    st.warning(translate("No test questions were generated for this vacancy. Please try another vacancy."))

if __name__ == "__main__":
    main() 