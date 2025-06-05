# app.py

import streamlit as st
from job_search_agent import JobSearchAgent, ResumeParserimport streamlit as st
from job_search_agent import JobSearchAgent, ResumeParser

st.set_page_config(page_title="Job Search AI Agent", layout="wide")
st.title("🚀 Job Search AI Agent")
st.sidebar.title("Navigation")

option = st.sidebar.selectbox(
    "Choose a feature:",
    ["Profile Setup", "Job Search", "Resume Analysis", "Interview Practice", "Career Insights"]
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = JobSearchAgent()

agent = st.session_state.agent

if option == "Profile Setup":
    st.header("👤 Setup Your Profile")
    name = st.text_input("Full Name")
    target_role = st.text_input("Target Job Role", "Software Developer")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
    skills_input = st.text_area("Skills (comma-separated)", "Python, JavaScript, React, SQL")

    if st.button("Create Profile"):
        skills = [skill.strip() for skill in skills_input.split(",")]
        agent.setup_user_profile(name, skills, experience, target_role)
        st.success("Profile created successfully!")

elif option == "Job Search":
    st.header("🔍 Search Jobs")
    query = st.text_input("Job Search Query", "python developer")
    location = st.text_input("Location", "us")
    count = st.slider("Number of Jobs", 5, 50, 10)

    if st.button("Search Jobs"):
        with st.spinner("Searching for jobs..."):
            jobs = agent.search_and_match_jobs(query, location, count)

        if jobs:
            st.success(f"Found {len(jobs)} jobs!")
            for i, job in enumerate(jobs[:10]):
                with st.expander(f"{job['title']} at {job['company']} - Match: {job.get('match_score', 'N/A')}%"):
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Salary:** {job['salary']}")
                    st.write(f"**Source:** {job['source']}")
                    st.write(f"**Description:** {job['description'][:200]}...")
                    if job.get('matched_skills'):
                        st.write(f"**Matching Skills:** {', '.join(job['matched_skills'])}")
                    st.write(f"[Apply Here]({job['url']})")
        else:
            st.warning("No jobs found. Try different search terms.")

elif option == "Resume Analysis":
    st.header("📄 Resume Analysis")
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])

    if uploaded_file is not None:
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.read())

        parser = ResumeParser()
        analysis = parser.parse_pdf_resume("temp_resume.pdf")

        if "error" not in analysis:
            st.success("Resume parsed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Resume Stats")
                st.write(f"Word Count: {analysis['word_count']}")
                st.write(f"Skills Found: {len(analysis['skills'])}")
                st.write(f"Emails: {analysis['emails']}")
                st.write(f"Phones: {analysis['phones']}")

            with col2:
                st.subheader("🛠️ Skills Detected")
                for skill in analysis['skills']:
                    st.markdown(f"- {skill}")
        else:
            st.error(analysis['error'])

elif option == "Interview Practice":
    st.header("🎭 Interview Practice")
    if agent.user_profile:
        job_title = st.text_input("Job Title for Practice", agent.user_profile.get('target_role', 'Developer'))
        if st.button("Generate Questions"):
            questions = agent.interview_prep.generate_interview_questions(job_title, agent.user_profile.get('skills', []))
            st.subheader("🤔 Behavioral Questions")
            for q in questions['behavioral'][:3]:
                st.write(f"• {q}")

            st.subheader("💻 Technical Questions")
            for q in questions['technical'][:3]:
                st.write(f"• {q}")

            st.subheader("🎯 Role-Specific Questions")
            for q in questions['role_specific']:
                st.write(f"• {q}")
    else:
        st.warning("Please setup your profile first!")

elif option == "Career Insights":
    st.header("📈 Career Insights")
    if agent.user_profile:
        insights = agent.generate_career_insights()
        if "error" not in insights:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Profile Summary")
                st.write(f"Experience Level: {insights['experience_level']}")
                st.write(f"Next Skills: {', '.join(insights['next_skills_to_learn'])}")
            with col2:
                st.subheader("📊 Skill Breakdown")
                for category, info in insights['skill_breakdown'].items():
                    st.write(f"{category}: {info['count']} skills ({info['percentage']}%)")
            st.subheader("💡 Recommendations")
            for rec in insights['recommendations']:
                st.write(f"• {rec}")
        else:
            st.error(insights['error'])
    else:
        st.warning("Please setup your profile first!")

