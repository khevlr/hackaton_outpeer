import os
from supabase import create_client, Client

from dotenv import load_dotenv
import json
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

new_vacancy = {
    "position_name": "Software Engineer",
    "summary": "We are looking for a software engineer with 3 years of experience in Python and Django.",
    "location": "San Francisco",
    "salary": "100000",
    "company_name": "Tech Corp",
    "skills": json.dumps(["Python", "Django"])
}

response = supabase.table("vacancies").insert(new_vacancy).execute()
print(response)