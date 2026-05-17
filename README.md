# SkillsFuture Course Recommender with LLM Gap Detection

Click here to view the [demo video](https://drive.google.com/file/d/1UadOzLvZQs_jdQr3SAw-R3Zju0krBBV5/view?usp=sharing)!


## How to run

### 1. Prerequisites

- Python `3.11+`
- `uv` installed
- (Optional) Docker Desktop
- Install Ollama from [Ollama’s Windows download page](https://ollama.com/download/windows)

    Ensure that you run the following:
    ```bash
    ollama pull llama3.2:3b
    ollama serve
    ```
### 2. Local run (recommended for development)

Install dependencies:

```bash
uv sync
```

If `uv.lock` is missing or outdated:

```bash
uv lock
uv sync
```

Create environment file:

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Build data/index (one-time or whenever source data changes):

```bash
python src/ingest_chroma.py
```

Run app:

```bash
streamlit run app.py
```

Then open: `http://localhost:8501`

### 3. Docker run

Create environment file:

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:3b
```

Build and run (will take awhile as data will be ingested into the chromadb):

```bash
docker compose --env-file .env up --build
```

App will be available at: `http://localhost:8501`

Note that due to latency issues due to model choice/availability, chatbot would take around 5-6 minutes to generate an output. If it exceeds timeout threshold, a fallback template would be used to show the results. 

## Repository Structure

```
├── .dockerignore                    # Excludes files from Docker build context
├── .env                             # Local environment variables (LLM provider/model/keys)
├── .gitignore                       # Files/folders ignored by Git
├── Dockerfile                       # Docker image definition for Streamlit app
├── docker-compose.yml               # Docker Compose service configuration
├── README.md                        # Project overview, setup, and run instructions
├── PROCESS.md                       # Project process notes/documentation
├── pyproject.toml                   # Project metadata and direct dependencies (uv)
├── uv.lock                          # Locked dependency graph for reproducible installs
├── requirements.txt                 # Pip dependency list used by Docker build
├── app.py                           # Main Streamlit application (chatbot pipeline)
├── chroma_db/                       # Persisted Chroma vector database files
├── data/                            # Input datasets and precomputed artifacts
│   ├── jd.csv                       # Job description dataset
│   ├── skillsfuture_courses.xlsx    # Course catalog dataset
│   ├── course_skills.json           # Precomputed extracted course skills
│   └── data.ipynb                   # Data exploration notebook
├── skill_extraction/                          # Experimental/alternate scripts
│   ├── skillner_extractor.py        # Skill extraction via SkillNer + SpaCy
│   ├── precompute_skills.py         # Precomputes job/course skills to JSON
│   └── llm_extractor.py             # Alternative LLM skill extraction logic
└── src/                             # Core application modules
    ├── chroma_store.py              # ChromaDB init/query helper utilities
    ├── ingest_chroma.py             # Builds and ingests Chroma collections
    ├── semantic_gap.py              # Semantic matching and gap logic
    ├── scoring.py                   # Course scoring and label mapping
    ├── llm_client.py                # LLM provider API wrapper
    ├── llm_answer.py                # Final answer prompt and response logic
    └── debug_chroma.py              # Debug helper for Chroma inspection
```

## Background 
### 1. What is the problem to solve?

SkillsFuture supports Singaporeans in reskilling and upskilling, especially as technology and AI reshape the labour market. However, course availability alone does not guarantee that workers are being trained for roles that employers actually need.

The problem is that SkillsFuture planners may find it difficult to quickly answer:

“For a given job role, what skills are required, what relevant courses are currently available, and what course gaps still exist?”

For example, if planners want to support people moving into roles such as data engineer, cybersecurity analyst, or AI engineer, they need to compare job requirements against existing course offerings. This is currently difficult because information is spread across job postings, course descriptions, skills frameworks, and labour-market dashboards.

There are also features available but it is only able to do a one to one comparison between job description and course overview. 

The project aims to build an AI-powered assistant that helps planners assess whether current SkillsFuture courses are relevant to specific career roles and identify what more needs to be done.



### 2. Who is affected by this problem?

The main stakeholder is SkillsFuture Singapore, especially teams involved in course planning, workforce development, and policy evaluation.



### 3. What is the impact of not solving this problem?

If this problem is not solved, Singapore may face a growing mismatch between the skills citizens have and the skills employers need. As AI and automation change job requirements, workers who are not equipped with relevant skills may find it harder to move into new or higher-demand roles.

There is also a risk that public upskilling resources are not used effectively. As training providers may continue offering generic courses that do not address important skill gaps.



### 4. How could data science or AI address this problem? Why is it necessary?

AI is useful because the key information is mostly unstructured. Job postings, course descriptions, and skills frameworks are written in natural language, making them hard to compare manually at scale.

The proposed solution is an AI-powered chatbot for SkillsFuture planners. A planner can enter a role, such as “data engineer”, and the system will:
Retrieve relevant job description.
Retrieve relevant SkillsFuture courses.
Compare course content against job requirements.
Identify missing skills or course gaps.
Suggest what new courses or course bundles could be introduced.



### 5. What would success look like?

Success means the tool helps SkillsFuture planners make better decisions about whether current courses are aligned with labour-market demand and citizen interest.

In the context of this prototype, success is measured by whether the system can correctly support the course-gap analysis workflow. This includes whether it can retrieve relevant job descriptions for a given role, identify the required skills from those job descriptions, retrieve relevant SkillsFuture courses, and flag skills that are weakly covered or not clearly covered by the current course catalogue.

The evaluation approach is described in more detail in the **Evaluation and Results** section below.


### 6. Current similar tools 
One relevant existing tool is SkillsFuture’s [Skills Extraction and Comparison Tool](https://jobsandskills.skillsfuture.gov.sg/data-and-tools/tools/skills-extraction-and-comparison-tool).

This tool uses machine learning and domain-adaptive language models to analyse unstructured text, such as job postings, course descriptions, or other free-text inputs. It can extract relevant skills from the text and compare two sets of text to identify shared and unique skills.

However, the tool is mainly designed for **one-to-one comparison**. For example, a user can compare one job description against one course outline to see which skills overlap and which skills are unique.

My project builds on a related idea but focuses on a broader planning use case. Instead of comparing only one job description to one course, the chatbot retrieves multiple relevant job descriptions for a role, compares them against multiple relevant retrieved courses, and identifies potential course-supply gaps.


## Dataset used

This project uses two main datasets: job descriptions and SkillsFuture course listings.

### Job Description Dataset

The job description data was obtained from a [Kaggle job skill set dataset](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set).

This dataset was chosen because it includes a `job_skill_set` column, which provides pre-labelled skills for each job description. This was useful for the prototype because I was unable to integrate SkillsFuture’s own [Skills Extraction and Comparison Tool](https://jobsandskills.skillsfuture.gov.sg/data-and-tools/tools/skills-extraction-and-comparison-tool) within the project constraints.

The job description dataset is used to represent labour-market demand. In the pipeline, relevant job descriptions are retrieved based on the user’s query, and their extracted skills are aggregated to form the required skill profile for a given role.

### SkillsFuture Course Dataset

The course data was collected from the [MySkillsFuture course search portal](https://www.myskillsfuture.gov.sg/content/portal/en/training-exchange/course-landing.html).

This dataset is used to represent the current course supply. 

Since the course dataset did not come with a clean structured skill list, I used [SkillNER](https://github.com/AnasAito/SkillNER) to extract skills from the course text as a temporary measure. This allowed me to create a course-skill inventory and test whether the overall gap-detection mechanism works.

### Data Limitation

The extracted course skills are not perfect. Some extracted skills are broad, noisy, or include soft skills. However, this was acceptable for the prototype because the aim was to test the end-to-end mechanism: retrieving job descriptions, identifying required skills, comparing them against course skills, and surfacing possible course gaps.

In a stronger production version, I would replace or improve the skill extraction step using a SkillsFuture-aligned taxonomy or official skills extraction tool, so that the extracted skills are more consistent with Singapore’s Skills Framework.


## Deployment Considerations

In a realistic deployment, this tool would likely be run internally by SkillsFuture Singapore or a related public-sector workforce planning team. It would be hosted as an internal dashboard or chatbot for officers involved in course planning, skills policy, and training-provider evaluation. The system should run in a secure government-approved cloud environment or internal analytics platform, with access limited to authorised users.

At realistic scale, the heavier processing should be done offline. Skill extraction, embedding generation, and ChromaDB index updates can run as scheduled batch jobs whenever new job postings or course data are added. The live application would mainly perform retrieval, similarity comparison, scoring, and optional answer generation. Since the embedding model used is lightweight, CPU inference should be sufficient for the prototype, although a production system may need stronger compute if processing large volumes of job postings regularly. 

Once live, I would monitor retrieval relevance, response latency, failed data ingestions, changes in detected course gaps. The risk that would keep me up at night is false confidence: the system may label a skill as “covered” because a course description contains similar wording, even though the course does not teach that skill deeply enough. For this reason, the tool should support human review before any funding or policy decision is made.


## Evaluation & Results

For the current prototype, evaluation was done manually by comparing the system output against the expected gaps for selected test queries. Since the current implementation uses a limited top-k retrieval setting due to model and runtime constraints, the evaluation is intended to show whether the mechanism works rather than to claim production-level accuracy.

For the query **“Data Analyst”**, I manually selected a set of expected skill gaps from the job descriptions and checked whether the system surfaced them in the final gap analysis. The expected gaps were:
- Problem solving
- Attention to detail
- Communication
- Data analysis
- Data management

These were chosen because they appeared in the job description skills but were not clearly represented in the extracted course-skill inventory. This made them useful as a small benchmark for checking whether the tool could identify skills that were required by jobs but weakly covered by the courses.

The table below shows whether each expected gap was detected across four runs. A value of `1` means the expected gap was surfaced by the system.

| Run | Problem solving | Attention to detail | Communication | Data analysis | Data management |
|---|---:|---:|---:|---:|---:|
| Run 1 | 1 | 1 | 1 | 1 | 1 |
| Run 2 | 1 | 1 | 1 | 1 | 0 |
| Run 3 | 1 | 0 | 1 | 0 | 1 |
| Run 4 | 1 | 1 | 1 | 1 | 1 |
| Run 5 | 1 | 1 | 1 | 1 | 1 |

Across the five runs, the system identified **22 out of 25 expected gap instances**, giving a manual gap-detection recall of **88%** for this small test case.

| Expected gap | Detected across runs |
|---|---:|
| Problem solving | 5 / 5 |
| Attention to detail | 4 / 5 |
| Communication | 5 / 5 |
| Data analysis | 4 / 5 |
| Data management | 4 / 5 |

This suggests that the prototype is able to surface skills that appear in job requirements but are weakly represented in the course-skill inventory. In particular, the tool consistently detected **problem solving** and **communication** as gaps across all five runs.

However, the current evaluation has clear limitations. Since only a limited number of top-k documents are retrieved, some relevant job descriptions or courses may be missed. The evaluation also depends on the quality of the extracted skills, which are noisy in the current dataset. Therefore, the current results should be interpreted as directional evidence that the pipeline works, not as a final measurement of production accuracy.

A stronger evaluation would use a larger top-k setting and a hand-labelled benchmark. For example, I would create ground-truth test scenarios by selecting or planting specific job descriptions and courses with known skill gaps. Each scenario would define:

- the target role
- the expected relevant job descriptions
- the courses that should be retrieved
- the specific skills that should be identified as covered or missing

The tool could then be evaluated on whether it retrieves the correct documents and whether it correctly identifies the planted skill gaps. This would make the evaluation more systematic and allow metrics such as retrieval precision, course relevance, and gap-detection accuracy to be reported.

For now, the manual evaluation shows that the prototype can demonstrate the intended workflow: retrieve relevant job descriptions, extract and aggregate required skills, compare them against course skills, and flag potential course gaps for human review.


## Current Limitations

This prototype is intended as a proof of concept, not a production-ready course-planning system. The current implementation demonstrates the end-to-end mechanism, but there are several limitations.

First, the current system uses a relatively small top-k retrieval setting. Due to model and runtime constraints, I lowered the number of retrieved job and course records so that the app can run smoothly while still showing how the pipeline works. This means the results may not reflect the full range of relevant job descriptions or courses. With stronger compute, a larger embedding model, or better indexing, I would increase the retrieval depth and compare against a wider set of courses.

Hence, the current comparison is still limited by top-k retrieval. Even though the app checks job skills against the course-skill inventory, the displayed course examples are still based on the most similar retrieved courses. A stronger production version should compare against more courses, support filtering by sector and skill level.

Second, the quality of the gap analysis depends heavily on skill extraction. The current extracted skills from both job descriptions and course descriptions are noisy. Some extracted items are soft skills, overly broad phrases, or terms that are not useful for technical course-gap analysis. In a stronger version, I would align the extraction step with SkillsFuture’s own skills taxonomy or tools such as SSG’s Skills Extraction Algorithm, which uses SSG’s skills dictionary and the national Skills Framework to identify skills from text. 

I would also add a feedback-tracking component inside the chatbot. After each response, users could rate whether the answer was useful or not, and optionally flag issues such as irrelevant retrieved job descriptions, irrelevant course matches, or gap analysis that does not make sense. These flagged responses could then be reviewed to identify recurring failure patterns, improve the skill extraction pipeline, adjust similarity thresholds, and refine the retrieval logic over time.

Third, the embeddings are affected by dataset quality. Ideally, the system would embed clean role descriptions and well-extracted skill fields. In the current dataset, some course overview fields are sparse or incomplete, so I embedded a fuller combination of available columns to avoid losing important context. This improves recall, but it also introduces more noise.

Fourth, latency is a limitation. The chatbot can take some time to load and generate results because it performs multiple steps: retrieving documents, comparing embeddings, checking course-skill coverage, scoring course examples, and optionally generating a final response using an LLM. Currently, the final generation step seems to be the bottle neck which i feel would scale efficiently with the use of a stronger model. 

Finally, this tool should not be treated as an automated decision-maker. It can surface possible gaps, but SkillsFuture planners, curriculum specialists, and training providers would still need to review the evidence before deciding whether to create, fund, or revise courses.

