import os
import numpy as np
import faiss
import g4f
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
import kagglehub
import pandas as pd
import asyncio

# Устанавливаем политику event loop для Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Отключаем предупреждение о симлинках
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Глобальные переменные
VECTOR_STORE_DIR = "vector_store"
DOCS_IN_RETRIEVER = 15
RELEVANCE_THRESHOLD_DOCS = 0.7
RELEVANCE_THRESHOLD_PROMPT = 0.6

# Инициализация локальных embeddings
print("Инициализация модели эмбеддингов...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}  # Используем CPU; замените на 'cuda' при наличии GPU
)
print("Модель эмбеддингов успешно инициализирована.")

# Функция для сохранения векторной базы
def save_vector_store(vector_store, vector_store_dir: str):
    print(f"Сохранение векторного хранилища в {vector_store_dir}...")
    vector_store.save_local(vector_store_dir)
    print(f"Vector store сохранён в: {vector_store_dir}")

# Функция для загрузки векторной базы
def load_vector_store(vector_store_dir: str, embeddings):
    print(f"Попытка загрузки векторного хранилища из {vector_store_dir}...")
    index_file = os.path.join(vector_store_dir, "index.faiss")
    if not os.path.exists(index_file):
        print(f"Файл {index_file} не найден. Не удалось загрузить vector store.")
        return None
    try:
        vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store успешно загружен из: {vector_store_dir}")
        return vector_store
    except Exception as e:
        print(f"Ошибка при загрузке vector store: {e}")
        return None

# Функция для загрузки и индексирования документов
def load_and_index_documents(vector_store_dir: str, embeddings) -> bool:
    print("Проверка наличия существующего векторного хранилища...")
    vector_store = load_vector_store(vector_store_dir, embeddings)
    if vector_store:
        print("Существующий vector store успешно загружен.")
        return True

    print("Загрузка датасета из Kaggle...")
    try:
        path = kagglehub.dataset_download("visualcomments/russian-civil-code-all-parts")
        print("Path to dataset files:", path)
        csv_file = os.path.join(path, "source.csv")
        if not os.path.exists(csv_file):
            print(f"Файл {csv_file} не найден.")
            return False

        df = pd.read_csv(csv_file)
        print("Датасет успешно загружен. Первые 5 записей:\n", df.head())
        if 'Contents' not in df.columns:
            print("Столбец 'Contents' не найден в датасете.")
            return False

        df['Contents'] = df['Contents'].fillna('').astype(str)
        loader = DataFrameLoader(df, page_content_column='Contents')
        documents = loader.load()

        if not documents:
            print("Не удалось загрузить документы из датасета.")
            return False
        print(f"Загружено {len(documents)} документов.")

        print("Разбиение документов на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        print(f"Всего получено {len(split_docs)} чанков после разбиения.")

        print("Индексация документов в FAISS...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        print("Документы успешно проиндексированы в FAISS.")

        save_vector_store(vector_store, vector_store_dir)
        return True
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return False

# Функция для предобработки запроса пользователя
def preprocess_user_prompt(user_prompt: str, chat_history: list) -> str:
    print("Предобработка пользовательского запроса...")
    instructions = (
        "Your task is to refine the user prompt below, preserving its meaning.\n"
        "Steps to follow:\n"
        "1. Identify the main question or request.\n"
        "2. If there are multiple tasks, list them.\n"
        "3. Keep the text concise and clear.\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        "Chat history:\n"
        f"{chat_history}\n"
        "-----\n"
        "Now, provide the improved prompt below:\n"
    )
    try:
        response = g4f.ChatCompletion.create(
            model="deepseek-v3",
            messages=[{"role": "system", "content": instructions}],
        )
        if isinstance(response, str):
            improved_prompt = response.strip()
        else:
            improved_prompt = response.choices[0].message.content.strip()
        print("Запрос успешно предобработан.")
    except Exception as e:
        print(f"Ошибка при предобработке запроса: {e}")
        improved_prompt = user_prompt
        print("Используется исходный запрос из-за ошибки.")
    return improved_prompt

# Функция для извлечения документов
def retrieve_documents(vector_store, user_prompt: str, k: int = 20):
    print(f"Извлечение до {k} релевантных документов...")
    if not vector_store:
        print("Vector store не загружен.")
        return []
    try:
        docs_with_scores = vector_store.similarity_search_with_score(user_prompt, k=k)
        print(f"Извлечено {len(docs_with_scores)} документов.")
        return docs_with_scores
    except Exception as e:
        print(f"Ошибка при извлечении документов: {e}")
        return []

# Функция для вычисления косинусной похожести
def compute_embeddings_similarity(embeddings, prompt: str, documents: list):
    print("Вычисление косинусной похожести документов...")
    if not documents:
        print("Нет документов для анализа.")
        return []
    try:
        prompt_embedding = np.array(embeddings.embed_query(prompt))
        relevance_scores = []
        for doc in documents:
            doc_embedding = np.array(embeddings.embed_query(doc.page_content))
            dot_product = np.dot(prompt_embedding, doc_embedding)
            norm_prompt = np.linalg.norm(prompt_embedding)
            norm_doc = np.linalg.norm(doc_embedding)
            similarity = 0.0
            if norm_prompt > 1e-9 and norm_doc > 1e-9:
                similarity = dot_product / (norm_prompt * norm_doc)
            similarity = np.clip(similarity, -1.0, 1.0)
            relevance_scores.append((doc, similarity))
        print(f"Косинусная похожесть вычислена для {len(relevance_scores)} документов.")
        return relevance_scores
    except Exception as e:
        print(f"Ошибка в compute_embeddings_similarity: {e}")
        return [(doc, 0.0) for doc in documents]

# Функция для проверки релевантности запроса
def is_prompt_relevant_to_documents(relevance_scores, relevance_threshold=RELEVANCE_THRESHOLD_PROMPT):
    print("Проверка релевантности запроса...")
    if not relevance_scores:
        print("Нет оценок релевантности.")
        return False
    max_similarity = max((sim for _, sim in relevance_scores), default=0.0)
    print(f"Максимальная похожесть: {max_similarity:.4f}, порог: {relevance_threshold}")
    return max_similarity >= relevance_threshold

# Функция для постобработки ответа LLM
def postprocess_llm_response(llm_response: str, user_prompt: str, context_str: str = "", references: dict = None, is_relevant: bool = False) -> tuple:
    print("Постобработка ответа LLM...")
    if references is None:
        references = {}
    if not is_relevant:
        references = {}
        context_str = ""
    prompt_references = (
        "You are an advanced language model tasked with providing a final, "
        "well-structured answer based on the given content.\n\n"
        "### Provided Data\n"
        f"LLM raw response:\n{llm_response}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Context:\n{context_str}\n\n"
        f"References:\n{references}\n\n"
        f"is_relevant: {is_relevant}\n"
        "-------------------------\n"
        "Please re-check clarity and, if references exist, list them at the end.\n"
        "Return the final improved answer now:\n"
    )
    try:
        response = g4f.ChatCompletion.create(
            model="deepseek-v3",
            messages=[{"role": "system", "content": prompt_references}],
        )
        if isinstance(response, str):
            final_answer = response.strip()
        else:
            final_answer = response.choices[0].message.content.strip()
        print("Ответ успешно постобработан.")
    except Exception as e:
        print(f"Ошибка при постобработке ответа: {e}")
        final_answer = llm_response
        print("Используется исходный ответ LLM из-за ошибки.")
    return final_answer, references

# Основная функция для генерации ответа
async def generate_response(prompt: str, chat_history=None):
    print("Начало генерации ответа...")
    if chat_history is None:
        chat_history = []

    success = load_and_index_documents(VECTOR_STORE_DIR, embeddings)
    if not success:
        print("Ошибка: Не удалось загрузить или создать векторное хранилище.")
        return "Failed to load or create vector store.", None
    vector_store = load_vector_store(VECTOR_STORE_DIR, embeddings)
    if not vector_store:
        print("Ошибка: Не удалось загрузить векторное хранилище.")
        return "Unable to load Vector Store.", None

    prepared_prompt = preprocess_user_prompt(prompt, chat_history)
    print(f"Подготовленный запрос: {prepared_prompt}")

    retrieved_docs_with_scores = retrieve_documents(vector_store, prepared_prompt, k=DOCS_IN_RETRIEVER)
    retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
    relevance_scores = compute_embeddings_similarity(embeddings, prepared_prompt, retrieved_docs)
    relevant_docs = [doc for (doc, similarity) in relevance_scores if similarity >= RELEVANCE_THRESHOLD_DOCS]

    print(f"Отфильтровано {len(relevant_docs)} релевантных документов.")
    print("Релевантные документы:")
    for i, doc in enumerate(relevant_docs, 1):
        content_preview = (doc.page_content[:100] + "...") if len(doc.page_content) > 100 else doc.page_content
        print(f"Документ {i}: {content_preview}")

    if not relevant_docs:
        print("Релевантные документы не найдены.")
        fallback_answer = "I couldn't find relevant information to answer your question."
        final_answer, _ = postprocess_llm_response(fallback_answer, prompt, "", None, False)
        return final_answer, None

    context_str = ""
    for doc in relevant_docs:
        content = doc.page_content or 'N/A'
        context_str += f"Content:\n{content}\n---\n"
    print("Контекст, переданный модели:")
    print(context_str)

    system_prompt = (
        "You are an expert on the Russian Civil Code. Provide a concise answer based on the context:\n"
        f"{context_str}\n"
        "--- End Context ---\n"
        "If the user question isn't fully answered in the provided context, "
        "use your best judgment while staying truthful.\n"
    )

    try:
        response = g4f.ChatCompletion.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prepared_prompt}
            ],
        )
        if isinstance(response, str):
            answer_text = response.strip()
        else:
            answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при вызове LLM: {e}")
        answer_text = "Ошибка при генерации ответа. Пожалуйста, попробуйте снова."

    is_relevant = is_prompt_relevant_to_documents(relevance_scores)
    references = {}
    for doc in relevant_docs:
        references.setdefault("Russian Civil Code Text", []).append(doc.page_content[:100] + "...")

    final_answer, processed_refs = postprocess_llm_response(answer_text, prompt, context_str, references, is_relevant)
    if is_relevant:
        final_text = final_answer + "\n---\nReferences: Russian Civil Code dataset"
        source_files = list(processed_refs.keys()) if processed_refs else None
    else:
        final_text = final_answer
        source_files = None

    return final_text, source_files

# Пример использования
if __name__ == "__main__":
    async def main():
        print("Запуск программы...")
        user_query = "Какие права есть у арендатора по Гражданскому кодексу РФ?"
        chat_hist = ["User: Привет!"]
        print(f"Обработка запроса: {user_query}")

        answer, sources = await generate_response(user_query, chat_hist)
        print("\nВОПРОС:\n", user_query)
        print("\nОТВЕТ ОТ LLM:\n", answer)
        print("\nИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:\n", sources)
        print("Программа завершена.")

    asyncio.run(main())