# TazaAir — AI Air Quality & Smog Shield Agent

## 1. Problem It Solves

During autumn and winter in Kazakhstan (especially in **Almaty**, **Astana**, **Ust-Kamenogorsk**, and **Karaganda**), severe temperature inversions trap dangerous smog (`PM2.5`, `PM10`, and `CO`) near the ground. Standard weather apps often say *"Sunny, +5°C"*, misleading people into opening windows or going for outdoor runs when the air is hazardous. Meanwhile, a regular LLM like ChatGPT cannot see real-time air pollution, wind dispersion, or keep a personal air diary.

**TazaAir** solves this local health problem by combining an LLM ("brain") with real-time environmental APIs, scientific health math, and local file storage ("hands"):
- Fetches live `PM2.5`, `PM10`, `CO`, and `US AQI` for any city or highland escape spot (`Medeu`, `Shymbulak`, `Kok-Zhailau`, `Burabay`).
- Checks live wind speed and precipitation to determine whether smog will blow away or stay trapped.
- Converts outdoor `PM2.5` exposure into the **Berkeley Earth scientific cigarette equivalent** ($22\,\mu\text{g/m}^3$ for 24 hours = **1 cigarette**) and gives clear window-ventilation and FFP2 mask advice.
- Saves and reads personal observations in a local `diary.json` file.
- Extracts a clean, validated **Pydantic** `AirSafetyCard` via an **LCEL chain** (`report_prompt | llm | parser`).

---

## 2. Tools (`@tool` Functions in `tools.py`)

| Tool Name | Type | Description |
| :--- | :--- | :--- |
| `get_air_quality(city: str)` | **Live API** (Open-Meteo Air Quality + Geocoding) | Fetches real-time `PM2.5`, `PM10`, carbon monoxide (`CO`), and `US AQI` for any city or mountain spot (e.g., Almaty, Astana, Medeu, Shymbulak). Includes error handling for unknown cities and network issues. |
| `get_weather_and_wind(city: str)` | **Live API** (Open-Meteo Weather) | Fetches live temperature, wind speed (`km/h`), humidity, and precipitation, and evaluates whether weak wind is trapping smog. |
| `calculate_cigarettes(pm25: float, hours_outside: float)` | **Custom Math & Health Logic** | Calculates exact cigarettes inhaled during outdoor exposure and per 24 hours using the Berkeley Earth formula (`PM2.5 / 22.0`), returning actionable window and FFP2 mask guidance. |
| `air_diary(action: str, city: str, note: str)` | **Local File Storage** (`diary.json`) | Saves timestamped air quality notes (`action="save"`) or reads saved history (`action="read"`) from a local JSON file. |

---

## 3. Model & LangChain Architecture

- **LLM Model:** `gemini-3.5-flash-lite` via **Google AI Studio** (`ChatGoogleGenerativeAI`), chosen for fast tool calling, native support for Gemini tool signatures, and generous free-tier limits.
- **Prompts:** `ChatPromptTemplate` with a structured multi-rule `SYSTEM_PROMPT`, `MessagesPlaceholder("chat_history")`, user input `{input}`, and `MessagesPlaceholder("agent_scratchpad")`.
- **Agent:** `create_tool_calling_agent` + `AgentExecutor` (`tools.py` + `agent.py`).
- **Memory:** `RunnableWithMessageHistory` backed by `InMemoryChatMessageHistory` per `session_id` for multi-turn follow-up questions.
- **LCEL + Structured Output:** `report_chain = report_prompt | llm | parser` using `PydanticOutputParser` and Pydantic `BaseModel` (`AirSafetyCard`).
- **Tracing:** **LangSmith** enabled via environment variables in `.env`.

---

## 4. How to Run the Project

1. Clone the repository and open the folder in **PyCharm** (or terminal):
   ```bash
   git clone https://github.com/ZeMe1/TazaAir.git
   cd TazaAir
   ```
2. Install dependencies inside your virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root (never committed to Git):
   ```env
   GEMINI_API_KEY=your_google_ai_studio_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=TazaAir
   LANGCHAIN_API_KEY=your_langsmith_api_key
   ```
4. Run `main.py`:
   ```bash
   python main.py
   ```
   - Select **`1`** for interactive live chat (type `card` anytime to generate a structured Pydantic `AirSafetyCard`).
   - Select **`2`** to automatically run all **3 example conversations** end-to-end.

---

## 5. Example Conversations

### Conversation 1 — Multi-Turn Smog Check, Follow-Up Math, Mountain Escape & Diary (Memory Test)

**Turn 1:**
> **User:** How is the air quality and wind in Almaty right now?
>
> `[Tools called: get_air_quality, get_weather_and_wind, calculate_cigarettes]`
>
> **TazaAir:** Here is the current air quality and weather report for **Almaty**:
> - **Air Quality (US AQI):** 75 (Moderate)
> - **PM2.5:** 24.4 µg/m³ | **PM10:** 25.6 µg/m³ | **CO:** 1003.0 µg/m³
> - **Weather & Wind:** 5.9 °C, 60% humidity, with a **very weak wind of 1.8 km/h**. Because the wind is so light, smog tends to accumulate and stay trapped in the basin.
> - **Cigarette Equivalent:** A full 24 hours outdoors at this level equals roughly **1.11 cigarettes/day**.
> - **Ventilation:** Brief window ventilation is acceptable, but avoid keeping windows open all day.
>
> ```text
> --- Structured Safety Card (LCEL + Pydantic) ---
> City:               Almaty
> Smog Status:        Moderate
> Cigarettes / 24h:   1.11
> Window Advice:      Brief window ventilation is acceptable, but avoid keeping windows open all day.
> FFP2 Mask Needed:   False
> ------------------------------------------------
> ```

**Turn 2 (Follow-up using chat memory — does not repeat city name or PM2.5 value):**
> **User:** If I walk outside there for 3 hours, how many cigarettes is that? Also check Medeu to see if the air is cleaner.
>
> `[Tools called: calculate_cigarettes, get_air_quality]`
>
> **TazaAir:**
> - **Almaty Walk (3 hours):** At 24.4 µg/m³, walking outside for 3 hours equals inhaling **0.14 cigarettes**.
> - **Medeu Air Quality:** Significantly cleaner up in the mountains — **US AQI: 39 (Good)**, **PM2.5: 3.0 µg/m³**, **CO: 112.0 µg/m³**. Heading up to Medeu is a great choice to escape trapped city smog!

**Turn 3 (Follow-up saving to local file):**
> **User:** Save a short note to my air diary comparing Almaty and Medeu today.
>
> `[Tools called: air_diary]`
>
> **TazaAir:** Saved to your air diary: `Almaty US AQI 75 (PM2.5 24.4), while Medeu air is much cleaner with US AQI 39 (PM2.5 3.0).`

---

### Conversation 2 — Astana Window Ventilation Check (Russian)

> **User:** Какой сейчас воздух и ветер в Астане? Можно ли открыть окно для проветривания?
>
> `[Tools called: get_air_quality, get_weather_and_wind, calculate_cigarettes]`
>
> **TazaAir:** В Астане сейчас отличные показатели воздуха и умеренный ветер:
> - **Качество воздуха (US AQI):** 25 (Чистый воздух)
> - **PM2.5:** 6.0 мкг/м³
> - **Погода и ветер:** 0.0 °C, влажность 92%, ветер **14.9 км/ч** (способствует хорошему рассеиванию).
> - **Вердикт:** Окно открывать можно и нужно — воздух чистый!
>
> ```text
> --- Structured Safety Card (LCEL + Pydantic) ---
> City:               Астана
> Smog Status:        Clean
> Cigarettes / 24h:   0.3
> Window Advice:      Да, окно открывать можно и нужно!
> FFP2 Mask Needed:   False
> ------------------------------------------------
> ```

---

### Conversation 3 — Reading Local Diary & Graceful Error Handling

> **User:** Show me all saved notes from my air diary, and also check air quality in WakandaForeverCity.
>
> `[Tools called: air_diary, get_air_quality]`
>
> **TazaAir:**
> - **Saved Diary Entry:** `[2026-11-25 23:41] Almaty: Almaty US AQI 75 (PM2.5 24.4), while Medeu air is much cleaner with US AQI 39 (PM2.5 3.0).`
> - **WakandaForeverCity:** Could not find location `'WakandaForeverCity'`. Please check the city name and try again.

---

## 6. LangSmith Tracing

LangSmith tracing is enabled via `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=TazaAir` in `.env`, recording every user prompt, tool selection, API output, memory injection, and Pydantic parser step.

- **LangSmith Project:** `TazaAir`
- **Public Shareable Trace Links:**
  1. [Conversation 1 (Turn 1) — Almaty Smog, Wind & Cigarette Equivalent (`RunnableWithMessageHistory` + 3 Tools)](https://smith.langchain.com/public/081a46f8-c05e-4fc7-931f-18954493fabc/r)
  2. [LCEL Structured Output Chain — `report_prompt | llm | parser` (`PydanticOutputParser` -> `AirSafetyCard`)](https://smith.langchain.com/public/c3f2378b-c813-4e7e-88ad-16f4ec1314c3/r)
  3. [Conversation 1 (Turn 2 Follow-up) — Memory Recall (`3h walk`) & Medeu Highland Comparison](https://smith.langchain.com/public/0929b636-b63d-4d06-9c92-3719dda3a05c/r)
  4. [Conversation 1 (Turn 3 Follow-up) — Saving Comparison to Local `diary.json`](https://smith.langchain.com/public/6804624f-108a-479d-99fb-23051a370416/r)
  5. [Conversation 2 — Astana Air & Window Ventilation Check (Russian)](https://smith.langchain.com/public/ed7dcca8-e9a9-4306-b67c-392296c25e7d/r)
  6. [Conversation 3 — Reading Local Diary & Graceful Error Handling (`WakandaForeverCity`)](https://smith.langchain.com/public/86febfa4-807e-476a-8381-3c40b115bd1a/r)
