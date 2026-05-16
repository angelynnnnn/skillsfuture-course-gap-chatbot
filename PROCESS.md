# Process

## Interface Design Decision

I chose a chatbot interface instead of a simple input-output form because the intended user is a planner who may want to ask exploratory questions. A fixed form could return a result for one role, but a chatbot better supports natural questions such as “What course gaps exist for data analysts?” or “Which skills are weakly covered by current courses?”

Ideally, this would become a chat-based RAG system where planners can ask questions about job descriptions, course offerings, required skills, and course gaps. The system would retrieve relevant evidence from the job and course data before answering.

For this prototype, I focused mainly on the gap extraction portion. The chatbot interface demonstrates how a planner might interact with the system, while the core work is in retrieving relevant job descriptions, aggregating required skills, comparing them against the course-skill inventory, and identifying possible gaps.


## Build Process and Design Decisions

When I first started building the prototype, my initial mechanism was quite simple. I wanted to generate embeddings for the job descriptions and the SkillsFuture course descriptions, then compare their similarity scores to identify whether the existing courses were relevant to a given job role.

This was the most straightforward approach because it allowed the system to compare job and course text directly. However, after testing and thinking through the use case more carefully, I realised that this approach was too noisy. A job description and a course description could be semantically similar at a broad level, but that did not necessarily mean the course covered the actual skills required by the job. For example, two documents could both be related to data roles, but one might focus on data engineering pipelines while the other focuses on dashboards or general analytics.

Because of this, I dropped the idea of relying mainly on full-document embedding similarity. It was not specific enough for a course-gap analysis tool. The output also would not be very explainable because it would only show that a course was “similar” to a job description, without clearly showing which skills were covered or missing.

I then pivoted the mechanism from document-level comparison to skill-level gap detection. Instead of asking, “How similar is this course to this job description?”, I reframed the problem as, “What skills are required by the retrieved job descriptions, and are those skills covered by the available courses?” This made the output more useful because the system could identify specific skills that were covered, weakly covered, or not clearly covered.

Another part of the process that changed was when skill extraction should happen. My original plan was to extract skills only after the user entered a query. The flow would have been: retrieve relevant job descriptions, extract skills from those job descriptions live, then compare the extracted skills against the course data. Conceptually, this made sense because the system would only extract skills from documents relevant to the user’s query.

However, I found that this was not practical for an interactive chatbot. Skill extraction took too long, especially when there were multiple job descriptions and course descriptions to process. This created latency and made the app feel slow. It was also inefficient because the same underlying course and job data would be processed repeatedly across different queries.

Because of this, I dropped live skill extraction during the query flow and moved skill extraction into a preprocessing step. The extracted skills are saved before the app runs, and the chatbot uses these precomputed skills during retrieval and gap analysis. This made the app faster and more reliable because the heavier processing is done upfront rather than during each user interaction.

For the skill extraction method, I first tried using an LLM. I thought this would be useful because LLMs are good at understanding unstructured text and identifying relevant concepts. However, using an LLM for extraction across the dataset was too slow, and the outputs were not always consistent enough for a repeatable pipeline.

After that, I looked for existing skill extraction tools online. I found that SkillsFuture has its own skills extraction-related tools, which would have been highly relevant because this project is based on SkillsFuture course data. Ideally, I would have used a SkillsFuture-aligned extraction approach so that the extracted skills would better match Singapore’s skills taxonomy and Skills Framework.

However, I was not able to integrate that tool smoothly within the  environment constraints of this prototype. As a practical alternative, I used SkillNER, an open-source skill extraction library, to create the initial extracted skills dataset. I treated this as a temporary measure rather than a perfect solution. It allowed me to move forward with building the pipeline while still keeping skill extraction as a central part of the system.

Ideally, the system would only embed selected high-signal columns, such as the course overview, listed skills, and “what will be taught” field. This would keep the embeddings cleaner and more focused on course-skill matching.

However, the dataset was inconsistent. Some course overviews were sparse or generic, while important information was sometimes found in the course name or other fields. Because of this, I embedded a fuller combination of available columns to avoid losing useful context. This improved recall, but the trade-off is that it may introduce more noise into the similarity matching.


## Where I Exercised Judgment

The biggest judgment call was changing the unit of analysis from documents to skills. A direct embedding comparison would have been easier to build, but it would not have answered the real planning question clearly. SkillsFuture planners need to know which skills are covered and which are missing, not just which courses are generally similar to a job description.

Another judgment call was moving skill extraction into preprocessing. Although live extraction sounded more dynamic, it was not a good trade-off because of latency. Since job and course datasets do not change with every user query, it made more sense to extract skills in advance and reuse them.

I also had to balance ideal tools against practical constraints. I wanted to use a SkillsFuture-aligned extraction method, but when I could not integrate it within the project constraints, I used SkillNER as a workable temporary alternative. This allowed me to continue building the full pipeline while being honest about the limitations of the extracted skills.

I also exercised judgment in how the course data was embedded. Ideally, clean skill fields would be enough, but the dataset was inconsistent. Embedding multiple relevant fields introduced some noise, but it improved the chance that the retriever would capture the main course context.

## What tools I used?

For the local LLM component, I used `llama3.2:3b` through Ollama. This model is used to generate the final written response after the retrieval, scoring, and gap detection steps have already been completed. I chose `llama3.2:3b` because it gave the best balance between response quality and local runtime among the models I tested. Smaller models such as `llama3.2:1b` were faster, but the responses were less reliable and did not always follow the intended structure as well. Larger models could potentially produce better summaries, but they were slower and less suitable for a lightweight Dockerised prototype running on local hardware.

For embeddings, I used `sentence-transformers/all-MiniLM-L6-v2`. This model was used to convert job descriptions, course descriptions, and extracted skills into vector representations so that the system could perform semantic retrieval and similarity comparison. I chose this model because it provides a practical balance between speed, model size, and semantic matching quality. Since latency was already a constraint in this project, I needed an embedding model that could support the full retrieval and gap-analysis pipeline without making the app too slow.

I also considered larger embedding models such as `all-mpnet-base-v2`, which may produce higher-quality embeddings. However, larger models are slower and heavier, which made them less suitable for this prototype. The goal was not to maximise model performance at all costs, but to build a working end-to-end system that could run locally and be Dockerised reliably.

One limitation of `all-MiniLM-L6-v2` is that it works best with short text, such as role names, skill phrases, and short descriptions. Ideally, the system would embed only clean, high-signal fields such as job roles, course titles, and extracted skills, which would better fit the model’s strengths and reduce noise.

However, the current dataset was inconsistent. Some course overviews were sparse or generic, while important context appeared in fields like the course name or “what will be taught”. Because of this, I embedded a fuller combination of available fields. This was less ideal, but it reduced the risk of missing important course context.

Overall, the model choices were guided by the constraints of the project. I prioritised models that were lightweight, locally runnable, and fast enough for a Dockerised prototype, while still being good enough to demonstrate the core mechanism of skill-based course-gap detection.







