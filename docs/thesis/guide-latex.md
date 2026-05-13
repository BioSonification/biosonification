Ниже — блок, который можно **добавить в единый документ с правилами ВКР**.

---

# 12. Как составлять ВКР в LaTeX на основе `nsu-diploma-template`

Для ВКР ФИТ НГУ можно использовать LaTeX-шаблон `shadrina/nsu-diploma-template`. В описании репозитория указано, что это шаблон пояснительной записки ФИТ НГУ, оформленный согласно пункту 2.4 программы ГИА. Сборка выполняется через `xelatex` или `lualatex`, а для библиографии дополнительно используется `bibtex`. ([GitHub][1])

## 12.1. Общая идея работы с шаблоном

LaTeX-шаблон устроен модульно: основной файл `main.tex` не содержит весь текст диплома сразу, а подключает отдельные файлы для титульного листа, задания, аннотации, содержания, введения, глав, заключения, списка источников и приложений. Это удобно: можно писать каждую часть ВКР в отдельном `.tex`-файле и не держать весь диплом в одном огромном документе. ([GitHub][2])

Структура подключения в `main.tex` выглядит логически так:

```latex
\documentclass[a4paper, 12pt]{extreport}
\usepackage{preamble}

\begin{document}

\input{data}
\input{title}
\input{task}
\input{abstract}
\input{toc}
\input{glossary}
\input{intro}
\input{chapter1}
\input{chapter2}
\input{chapter3}
\input{conclusion}
\input{bibliography}
\input{appendix1}

\end{document}
```

То есть при написании ВКР лучше работать не с одним файлом, а с набором файлов:

```text
main.tex          — главный файл, который всё собирает
data.tex          — данные студента, темы, кафедры, руководителя
title.tex         — титульный лист
task.tex          — задание на ВКР
abstract.tex      — аннотация
toc.tex           — содержание
glossary.tex      — определения, обозначения и сокращения
intro.tex         — введение
chapter1.tex      — первая глава
chapter2.tex      — вторая глава
chapter3.tex      — третья глава
conclusion.tex    — заключение
bibliography.tex  — список источников
biblio.bib        — база библиографических источников
appendix1.tex     — приложения
preamble.sty      — настройки оформления
```

---

## 12.2. Что сначала нужно заполнить

Первым делом нужно открыть файл `data.tex`. В нём задаются общие переменные: ФИО студента, группа, кафедра, направление подготовки, тема работы, университет, факультет, данные заведующего кафедрой, руководителя и соруководителя. Эти данные потом автоматически используются на титульном листе, в задании и аннотации. ([GitHub][3])

Пример того, что нужно заменить:

```latex
\newcommand{\studentlastname}{Фамилия}
\newcommand{\studentinitials}{Имя Отчество}
\newcommand{\group}{12345}
\newcommand{\department}{Название кафедры}
\newcommand{\specialization}{Название направления}
\newcommand{\topic}{Название вашей выпускной квалификационной работы}
```

Для своей ВКР нужно заменить шаблонные значения на реальные:

```latex
\newcommand{\studentlastname}{Власенко}
\newcommand{\studentinitials}{Иван ...}
\newcommand{\group}{22214}
\newcommand{\department}{Кафедра систем информатики}
\newcommand{\specialization}{Компьютерные науки и системотехника}
\newcommand{\topic}{Разработка модели генерации музыкальных последовательностей на основе биологических данных}
```

---

## 12.3. Что писать в каких файлах

Рекомендуемый порядок заполнения:

| Файл             | Что в нём писать                                             |
| ---------------- | ------------------------------------------------------------ |
| `data.tex`       | Все общие сведения: ФИО, группа, тема, кафедра, руководитель |
| `abstract.tex`   | Аннотация, ключевые слова, краткое описание результата       |
| `glossary.tex`   | Термины, сокращения, обозначения                             |
| `intro.tex`      | Введение                                                     |
| `chapter1.tex`   | Анализ предметной области и существующих решений             |
| `chapter2.tex`   | Проектирование и реализация                                  |
| `chapter3.tex`   | Проверка, эксперименты, оценка достоверности                 |
| `conclusion.tex` | Заключение                                                   |
| `biblio.bib`     | Источники в формате BibTeX                                   |
| `appendix1.tex`  | Приложения                                                   |

Главный файл `main.tex` обычно не нужно сильно менять. Его задача — собрать все части в один PDF.

---

## 12.4. Как добавить или убрать главу

Если нужна новая глава, например четвёртая, создаётся файл:

```text
chapter4.tex
```

Внутри него:

```latex
\chapter{Оценка экономической эффективности}

Текст главы...
```

Затем в `main.tex` нужно добавить строку:

```latex
\input{chapter4}
```

Например:

```latex
\input{chapter1}
\input{chapter2}
\input{chapter3}
\input{chapter4}
\input{conclusion}
```

Если какая-то часть не нужна, её можно временно закомментировать:

```latex
% \input{appendix1}
```

---

## 12.5. Настройки оформления уже вынесены в `preamble.sty`

Файл `preamble.sty` содержит основные настройки оформления: русский и английский языки, шрифт Times New Roman, поля страницы, межстрочный интервал, абзацный отступ, гиперссылки, изображения, таблицы, листинги, подписи к рисункам и таблицам, математику, содержание и оформление заголовков. В шаблоне поля заданы как левое 30 мм, правое 10 мм, верхнее и нижнее 20 мм, а основной шрифт — Times New Roman. ([GitHub][4])

Обычно файл `preamble.sty` лучше не трогать без необходимости. В нём уже настроены:

```latex
\setmainfont{Times New Roman}
\usepackage[left=30mm,right=10mm,top=20mm,bottom=20mm]{geometry}
\setlength\parindent{12.5mm}
```

То есть шаблон уже частично соответствует требованиям оформления ВКР.

---

## 12.6. Как писать главы

В LaTeX структура задаётся командами:

```latex
\chapter{Название главы}
\section{Название раздела}
\subsection{Название подраздела}
```

Пример для первой главы:

```latex
\chapter{Анализ предметной области и существующих решений}

\section{Описание предметной области}

Текст раздела...

\section{Анализ существующих решений}

Текст раздела...

\section{Недостатки существующих решений}

Текст раздела...

\section{Постановка задачи}

Текст раздела...
```

Для твоей ВКР можно использовать такую структуру:

```latex
\chapter{Анализ предметной области и существующих решений}

\section{Описание задачи генерации музыки на основе биологических данных}

\section{Анализ существующих подходов к сонфикации биологических данных}

\section{Анализ методов представления музыкальных последовательностей}

\section{Анализ моделей генерации музыки}

\section{Недостатки существующих решений}

\section{Постановка задачи}
```

---

## 12.7. Как вставлять рисунки

В шаблоне уже подключён пакет `graphicx`, а папка для изображений задана как `graphics`. Поэтому изображения удобно складывать в папку:

```text
graphics/
```

Пример вставки рисунка из шаблона:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.33\textwidth]{sample.jpg}
    \caption{Пример картинки}
    \label{fig:figure-sample}
\end{figure}
```

Для своей работы:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{architecture.png}
    \caption{Архитектура разработанной системы}
    \label{fig:architecture}
\end{figure}
```

Ссылка на рисунок в тексте:

```latex
Архитектура разработанной системы представлена на рисунке~\ref{fig:architecture}.
```

В шаблоне подписи к рисункам настроены через `caption`, поэтому они автоматически оформляются как «Рисунок ... — ...». ([GitHub][4])

---

## 12.8. Как вставлять таблицы

В шаблоне для таблиц используются пакеты `multirow` и `longtable`. В примере `chapter2.tex` показана таблица с подписью, меткой и окружением `longtable`; также в комментарии указано, что таблицы удобно генерировать через `tablesgenerator.com`. ([GitHub][5])

Простой пример таблицы:

```latex
\begin{longtable}{|p{0.25\textwidth}|p{0.3\textwidth}|p{0.35\textwidth}|}
\caption{Сравнение существующих решений}
\label{tab:solutions}\\
\hline
Решение & Преимущества & Недостатки \\
\hline
Подход 1 & Простота реализации & Низкая выразительность \\
\hline
Подход 2 & Высокая гибкость & Сложность настройки \\
\hline
\end{longtable}
```

Ссылка в тексте:

```latex
Сравнение существующих решений приведено в таблице~\ref{tab:solutions}.
```

---

## 12.9. Как вставлять формулы

В шаблоне подключены математические пакеты `amsthm`, `amssymb`, `amsmath`, `mathtools`, а в `chapter3.tex` показаны примеры формул через окружение `equation` с автоматической нумерацией и ссылкой через `\label` / `\ref`. ([GitHub][4]) ([GitHub][6])

Пример:

```latex
\begin{equation}
    H(X) = - \sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
    \label{eq:shannon-entropy}
\end{equation}
```

Ссылка в тексте:

```latex
Энтропия последовательности вычисляется по формуле~\ref{eq:shannon-entropy}.
```

---

## 12.10. Как вставлять листинги кода

В шаблоне подключён пакет `listings`, настроен стиль `mystyle`, включены номера строк, перенос строк и подписи. В `chapter3.tex` приведён пример листинга Java-кода с подписью. ([GitHub][4]) ([GitHub][6])

Пример для Python:

```latex
\begin{lstlisting}[language=Python, caption=Функция вычисления энтропии последовательности]
import math
from collections import Counter

def entropy(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())
\end{lstlisting}
```

Если код большой, его лучше вынести в приложение, а в основной части оставить только ключевые фрагменты.

---

## 12.11. Как оформлять источники

Список источников в шаблоне собирается через BibTeX. В файле `bibliography.tex` используется стиль `utf8gost71u`, подключается база `biblio.bib`, а сам раздел добавляется в содержание как «Список использованных источников и литературы». ([GitHub][7])

Пример источника в `biblio.bib`:

```bibtex
@article{musegan,
  author = {Dong, Hao-Wen and Hsiao, Wen-Yi and Yang, Li-Chia and Yang, Yi-Hsuan},
  title = {MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2018}
}
```

Ссылка в тексте:

```latex
Одним из известных подходов к генерации символической музыки является MuseGAN~\cite{musegan}.
```

После этого источник автоматически попадёт в список литературы.

---

## 12.12. Как собрать PDF

Минимальная схема сборки:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Или через `lualatex`:

```bash
lualatex main.tex
bibtex main
lualatex main.tex
lualatex main.tex
```

Повторный запуск `xelatex`/`lualatex` нужен, чтобы корректно обновились содержание, ссылки, номера рисунков, таблиц, формул и список литературы.

---

## 12.13. Как работать через Overleaf

В README репозитория указано, что пример итогового PDF можно посмотреть в Overleaf. Это значит, что шаблон можно использовать не только локально, но и как онлайн-проект: загрузить файлы в Overleaf, выбрать компилятор `XeLaTeX` или `LuaLaTeX` и собирать PDF прямо в браузере. ([GitHub][1])

Удобный порядок:

```text
1. Скачать репозиторий с шаблоном.
2. Загрузить его в Overleaf как новый проект.
3. Установить компилятор XeLaTeX.
4. Заполнить data.tex.
5. Писать главы в chapter1.tex, chapter2.tex, chapter3.tex.
6. Добавлять источники в biblio.bib.
7. Периодически собирать PDF и проверять оформление.
```

---

## 12.14. Практический порядок написания ВКР в LaTeX

Лучше писать не сверху вниз, а по инженерной логике:

```text
1. Заполнить data.tex.
2. Собрать пустой шаблон и убедиться, что PDF создаётся.
3. Составить структуру глав в chapter1.tex, chapter2.tex, chapter3.tex.
4. Написать введение черновиком.
5. Написать главу 1: анализ предметной области и аналогов.
6. Написать главу 2: проектирование и реализация.
7. Написать главу 3: проверка и оценка достоверности.
8. Добавить рисунки, таблицы, формулы и листинги.
9. Заполнить biblio.bib и расставить \cite{}.
10. Написать заключение.
11. Написать аннотацию.
12. Проверить содержание, ссылки, нумерацию и список источников.
```

---

## 12.15. Мини-чек-лист по LaTeX перед сдачей

```text
[ ] PDF собирается без критических ошибок.
[ ] В data.tex заполнены реальные данные.
[ ] Тема на титульном листе совпадает с утверждённой темой.
[ ] Содержание обновлено и номера страниц корректные.
[ ] Все рисунки имеют \caption{} и \label{}.
[ ] На все рисунки есть ссылки через \ref{}.
[ ] Все таблицы имеют \caption{} и \label{}.
[ ] На все таблицы есть ссылки через \ref{}.
[ ] Все формулы, на которые есть ссылки, имеют \label{}.
[ ] Все источники добавлены в biblio.bib.
[ ] Все источники из списка литературы реально цитируются через \cite{}.
[ ] Нет знаков вопроса вместо ссылок: ??.
[ ] Нет пустых шаблонных значений вроде “Фамилия”, “Название кафедры”.
[ ] Аннотация не превышает одну страницу.
[ ] Приложения подключены только при необходимости.
```

---

## 12.16. Главное правило работы с шаблоном

`main.tex` — это сборщик диплома.
`data.tex` — это паспортные данные работы.
`preamble.sty` — это оформление.
`chapter1.tex`, `chapter2.tex`, `chapter3.tex` — это основной текст ВКР.
`biblio.bib` — это источники.

То есть писать диплом в LaTeX нужно не путём ручного форматирования каждой страницы, а через структуру:

```text
данные → главы → рисунки/таблицы/формулы → источники → автоматическая сборка PDF
```

[1]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/README.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/main.tex "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/data.tex "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/preamble.sty "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/chapter2.tex "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/chapter3.tex "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/shadrina/nsu-diploma-template/master/bibliography.tex "raw.githubusercontent.com"
