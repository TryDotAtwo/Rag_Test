import asyncio
import logging
import os
import re
from typing import List, Tuple, Dict

import kagglehub
import pandas as pd
import g4f

# Установка политики событийного цикла для Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Порог для длинных статей (символы), учитывая вопрос и промпт
ARTICLE_LENGTH_THRESHOLD = 1000
# Максимальная длина группы коротких статей
MAX_GROUP_LENGTH = 2000
# Таймаут для запросов к g4f (секунды)
REQUEST_TIMEOUT = 60

def load_civil_code_dataset() -> Dict[int, str]:
    """
    Загружает датасет с Kaggle, парсит статьи ГК РФ из source.csv.
    Возвращает словарь {номер_статьи: текст_статьи}.
    """
    logger.info("Загрузка датасета из Kaggle...")
    try:
        path = kagglehub.dataset_download("visualcomments/russian-civil-code-all-parts")
        logger.info("Путь к загруженным файлам: %s", path)

        csv_file = os.path.join(path, "source.csv")
        if not os.path.exists(csv_file):
            logger.error("Файл %s не найден.", csv_file)
            raise FileNotFoundError(f"Файл {csv_file} не найден.")

        df = pd.read_csv(csv_file)
        logger.info("Датасет успешно загружен.")
        logger.debug("Первые 5 строк датасета:\n%s", df.head(5))

        if 'Names' not in df.columns or 'Contents' not in df.columns:
            logger.error("В датасете отсутствуют столбцы 'Names' или 'Contents'.")
            raise KeyError("В датасете отсутствуют столбцы 'Names' или 'Contents'.")

        dataset = {}
        for index, row in df.iterrows():
            name = str(row['Names'])
            content = row['Contents']

            if pd.isna(content):
                logger.debug("Пропущена строка с NaN в Contents: %s", name)
                continue

            match = re.match(r'Статья\s+(\d+)\.\s*(.*)', name)
            if not match:
                logger.debug("Строка не является статьей: %s", name)
                continue

            num = int(match.group(1))
            article_text = str(content).strip()

            if not article_text or article_text == 'nan':
                logger.warning("Пустой текст для статьи %d.", num)
                continue

            dataset[num] = article_text
            logger.debug("Добавлена статья %d: %s...", num, article_text[:50])

        if not dataset:
            logger.warning("Статьи не найдены в датасете.")
        else:
            logger.info("Загружено %d статей ГК РФ.", len(dataset))

        return dataset

    except Exception as e:
        logger.error("Ошибка при загрузке датасета: %s", e)
        raise

async def rephrase_question(question: str, question_label: str) -> str:
    """
    Переформулирует вопрос для лучшего понимания LLM.
    """
    prompt = (
        f"Переформулируй вопрос для лучшего понимания машиной. Верни только один переформулированный вопрос, "
        f"краткий и юридически точный, без дополнительных вариантов или пояснений.\n"
        f"Вопрос: \"{question}\"\n"
        f"Пример:\n"
        f"Вопрос: \"Что такое право собственности?\"\n"
        f"Ответ: \"Как определяется право собственности в гражданском праве?\"\n"
    )
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model="deepseek-v3",
                messages=[{"role": "user", "content": prompt}]
            )
        rephrased = response.strip()
        logger.info("[%s] Вопрос переформулирован: %s", question_label, rephrased)
        return rephrased
    except Exception as e:
        logger.error("[%s] Ошибка при переформулировке вопроса: %s", question_label, e)
        return question

async def find_relevant_articles(question: str, question_label: str) -> List[Tuple[int, str]]:
    """
    Находит релевантные статьи и кодекс.
    Возвращает список кортежей [(номер_статьи, кодекс)].
    """
    prompt = (
        f"По вопросу: \"{question}\", укажи номера статей и кодекс, к которому они относятся. "
        f"Верни ответ строго в формате '[Кодекс: номер1, номер2, ...]', без пояснений.\n"
        f"Пример:\n"
        f"Вопрос: \"Кто отвечает за вред от источника повышенной опасности?\"\n"
        f"Ответ: [Гражданский кодекс РФ: 1079, 1064]\n"
    )
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model="deepseek-v3",
                messages=[{"role": "user", "content": prompt}]
            )
        logger.debug("[%s] Ответ LLM на поиск статей: %s", question_label, response)

        match = re.match(r'\[(.*?):\s*([\d,\s]*)\]', response)
        if match:
            code = match.group(1).strip()
            nums_str = match.group(2).strip()
            nums = [int(n) for n in re.findall(r'\d+', nums_str)] if nums_str else []
        else:
            code_match = re.search(r'(Гражданский кодекс РФ|ГК РФ)', response)
            code = code_match.group(1) if code_match else "Гражданский кодекс РФ"
            nums = [int(n) for n in re.findall(r'\b\d+\b', response) if int(n) in range(1, 2000)]

        unique_nums = sorted(set(nums))
        articles = [(num, code) for num in unique_nums]
        logger.info("[%s] Найденные статьи: %s", question_label, articles)
        return articles
    except Exception as e:
        logger.error("[%s] Ошибка при поиске статей: %s", question_label, e)
        return []

async def generate_initial_answer(question: str, question_label: str) -> str:
    """
    Генерирует первоначальный ответ на вопрос.
    """
    prompt = (
        f"Дай краткий и точный ответ на вопрос: \"{question}\". Сфокусируйся на юридической основе, "
        f"избегай лишних деталей.\n"
        f"Пример:\n"
        f"Вопрос: \"Что такое право собственности?\"\n"
        f"Ответ: Право собственности — это право владеть, пользоваться и распоряжаться имуществом (ст. 209 ГК РФ).\n"
    )
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model="deepseek-v3",
                messages=[{"role": "user", "content": prompt}]
            )
        answer = response.strip()
        logger.info("[%s] Сгенерирован начальный ответ: %s", question_label, answer[:100] + "..." if len(answer) > 100 else answer)
        return answer
    except Exception as e:
        logger.error("[%s] Ошибка при генерации ответа: %s", question_label, e)
        return "Не удалось сгенерировать ответ."

async def refine_answer_with_article(question: str, current_answer: str, article_text: str, article_nums: List[int], question_label: str) -> str:
    """
    Уточняет ответ на основе текста статьи или группы статей.
    article_nums — номера статей для логирования.
    """
    if not article_text:
        return current_answer
    prompt = (
        f"Вопрос: \"{question}\"\n"
        f"Текст статьи(ей) ГК РФ: \"{article_text}\"\n"
        f"Текущий ответ: \"{current_answer}\"\n"
        f"Проанализируй текст статьи и текущий ответ. Сформируй новый ответ, который:\n"
        f"- Уточняет или дополняет текущий ответ с учетом статьи.\n"
        f"- Остается кратким и юридически точным.\n"
        f"- Ссылается на статью, если это уместно.\n"
        f"Верни только новый ответ, без пояснений.\n"
        f"Пример:\n"
        f"Вопрос: \"Кто отвечает за вред от источника повышенной опасности?\"\n"
        f"Текущий ответ: \"Отвечает владелец источника.\"\n"
        f"Текст статьи: \"Статья 1079: Юридические лица и граждане, деятельность которых связана с повышенной опасностью, обязаны возместить вред...\"\n"
        f"Ответ: Владелец или законный пользователь источника повышенной опасности обязан возместить вред (ст. 1079 ГК РФ).\n"
    )
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model="deepseek-v3",
                messages=[{"role": "user", "content": prompt}]
            )
        new_answer = response.strip()
        logger.info("[%s] Ответ уточнен для статей %s: %s", question_label, article_nums, new_answer[:100] + "..." if len(new_answer) > 100 else new_answer)
        return new_answer
    except Exception as e:
        logger.error("[%s] Ошибка при уточнении ответа: %s", question_label, e)
        return current_answer

def group_articles(
    articles: List[Tuple[int, str, str]], question: str, prompt_template: str, question_label: str
) -> List[Tuple[List[int], str]]:
    """
    Группирует статьи для обработки: короткие объединяет, длинные разбивает.
    articles: [(номер_статьи, кодекс, текст_статьи)]
    Возвращает: [(номера_статей, текст_группы)]
    """
    grouped = []
    current_group = []
    current_text = ""
    prompt_length = len(prompt_template.format(question=question, article_text="", current_answer=""))

    for num, code, text in articles:
        article_length = len(text)
        total_length = prompt_length + article_length + len(current_text)

        if article_length + prompt_length > ARTICLE_LENGTH_THRESHOLD:
            if current_group:
                grouped.append((current_group, current_text))
                current_group = []
                current_text = ""
            chunks = [text[i:i+ARTICLE_LENGTH_THRESHOLD] for i in range(0, len(text), ARTICLE_LENGTH_THRESHOLD)]
            for chunk in chunks:
                grouped.append(([num], chunk))
        elif total_length <= MAX_GROUP_LENGTH:
            current_group.append(num)
            current_text += f"\nСтатья {num}: {text}\n" if current_text else f"Статья {num}: {text}"
        else:
            grouped.append((current_group, current_text))
            current_group = [num]
            current_text = f"Статья {num}: {text}"

    if current_group:
        grouped.append((current_group, current_text))

    logger.debug("[%s] Сгруппированные статьи: %s", question_label, [(nums, text[:50] + "..." if text else "") for nums, text in grouped])
    return grouped

async def process_question(question: str, dataset: Dict[int, str], question_label: str, debug: bool = False) -> Tuple[str, List[Tuple[int, str]], List[int]]:
    """
    Обрабатывает один вопрос: переформулирует, ищет статьи, генерирует и уточняет ответ.
    Возвращает: (ответ, использованные_статьи, отсутствующие_статьи).
    """
    if not question or not isinstance(question, str):
        logger.error("[%s] Вопрос пустой или некорректный.", question_label)
        return "Вопрос некорректен.", [], []

    logger.info("[%s] Обработка вопроса: %s", question_label, question)

    rephrased = await rephrase_question(question, question_label)
    if debug:
        logger.debug("[%s] Переформулированный вопрос: %s", question_label, rephrased)

    article_nums_codes = await find_relevant_articles(rephrased if rephrased else question, question_label)
    if debug:
        logger.debug("[%s] Найденные статьи: %s", question_label, article_nums_codes)

    articles = []
    missing_articles = []
    for num, code in article_nums_codes:
        text = dataset.get(num)
        if text:
            articles.append((num, code, text))
            logger.info("[%s] Найдена статья %d: %s...", question_label, num, text[:50])
        else:
            missing_articles.append(num)
            logger.warning("[%s] Статья %d ГК РФ не найдена.", question_label, num)

    prompt_template = (
        "Вопрос: \"{question}\"\n"
        "Текст статьи(ей) ГК РФ: \"{article_text}\"\n"
        "Текущий ответ: \"{current_answer}\"\n"
        "Проанализируй текст статьи и текущий ответ. Сформируй новый ответ, который:\n"
        "- Уточняет или дополняет текущий ответ с учетом статьи.\n"
        "- Остается кратким и юридически точным.\n"
        "- Ссылается на статью, если это уместно.\n"
        "Верни только новый ответ, без пояснений.\n"
        "Пример:\n"
        "Вопрос: \"Кто отвечает за вред от источника повышенной опасности?\"\n"
        "Текущий ответ: \"Отвечает владелец источника.\"\n"
        "Текст статьи: \"Статья 1079: Юридические лица и граждане, деятельность которых связана с повышенной опасностью, обязаны возместить вред...\"\n"
        "Ответ: Владелец или законный пользователь источника повышенной опасности обязан возместить вред (ст. 1079 ГК РФ).\n"
    )
    grouped_articles = group_articles(articles, question, prompt_template, question_label)
    if debug:
        logger.debug("[%s] Сгруппированные статьи: %s", question_label, [(nums, text[:50] + "..." if text else "") for nums, text in grouped_articles])

    answer = await generate_initial_answer(question, question_label)
    if debug:
        logger.debug("[%s] Начальный ответ: %s", question_label, answer)

    for article_nums, text in grouped_articles:
        answer = await refine_answer_with_article(question, answer, text, article_nums, question_label)
        if debug:
            logger.debug("[%s] Уточненный ответ после статей %s: %s", question_label, article_nums, answer[:100] + "..." if len(answer) > 100 else answer)

    return answer, article_nums_codes, missing_articles

async def main(questions_dict: Dict[str, str], debug: bool = False):
    """
    Основная функция для обработки списка вопросов.
    """
    if not questions_dict:
        logger.error("Список вопросов пуст.")
        return

    dataset = load_civil_code_dataset()
    if not dataset:
        logger.error("Датасет не загружен.")
        return

    tasks = [
        asyncio.create_task(process_question(question, dataset, label, debug))
        for label, question in questions_dict.items()
    ]
    results = await asyncio.gather(*tasks)

    logger.info("\n=== Результаты ===")
    for (answer, articles, missing_articles), (label, question) in zip(results, questions_dict.items()):
        logger.info("[%s] Вопрос: %s", label, question)
        logger.info("[%s] Ответ: %s", label, answer)
        if articles:
            code_articles = {}
            for num, code in articles:
                if code not in code_articles:
                    code_articles[code] = []
                code_articles[code].append(num)
            articles_str = "; ".join(f"{code}: {', '.join(map(str, sorted(nums)))}" for code, nums in code_articles.items())
            logger.info("[%s] Использованные статьи: %s", label, articles_str)
        else:
            logger.info("[%s] Использованные статьи: нет", label)
        if missing_articles:
            missing_str = ", ".join(map(str, sorted(missing_articles)))
            logger.warning("[%s] Не найдены статьи: %s", label, missing_str)

if __name__ == "__main__":
    questions_dict = {
        "q1": "Что такое право собственности?",
        "q2": "Какие бывают виды обязательств?",
        "q3": "Как определяется цена договора в гражданском праве?",
        "q4": "Кто отвечает за вред, причинённый источником повышенной опасности?",
        "q5": "Ответственность за действительность прав, удостоверенных документарной ценной бумагой",
    }

    asyncio.run(main(questions_dict, debug=True))