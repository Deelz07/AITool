from google import genai
from pydantic import BaseModel
import os
import polars as pl
from pathlib import Path

# topic = input("Choose a topic you would like to get information for (separate topics with commas):")
# prompt = "What is the largest country by landmass"

data_path = Path("data.parquet")



client = genai.Client(api_key=os.environ.get("GENAI_API_KEY") )

def generate_response(topics:str,client):
    prompt = f"""Generate a study database for:
    {topic}. If there are multiple topics make a row each topic.
    Make sure that at least five different items are generated for each category and please be detailed. 
    Separate each item with a ;.
    For all resources (Text and video) pleast attach a link
    """

    response = client.models.generate_content(model = "gemini-2.5-flash",
                                   contents = prompt,
                                   config = {'response_mime_type':'application/json',
                                             'response_schema':list[Module]})

    
    pass

def convert_to_markdown(df:pl.DataFrame):
    with open("STUDY_GUIDE.md", "w") as f:
        for row in df.to_dicts():
            f.write(f"# Topic: {row['topic']}\n\n")

            # 1. Prerequisites (Numbered List)
            f.write("### 🧠 Foundational Knowledge\n")
            # Split by semicolon and write as a numbered list
            prereqs = [p.strip() for p in row['prerequisites'].split(';') if p.strip()]
            for i, item in enumerate(prereqs, 1):
                f.write(f"{i}. {item}\n")
            f.write("\n")
            
            # 2. Learning Objectives (Checklist)
            f.write("## 🎯 Learning Objectives\n")
            for sub in row['sub_topics'].split(';'):
                    f.write(f"- [ ] Master **{sub.strip()}** \n \n ")
                
            # 3. Coding Lab (Checklist with bold titles)
            f.write("\n## 💻 Coding Lab\n")
            # Assuming projects are separated by ';' in your database
            for project in row['coding_projects'].split(';'):
                    # Reformat "Name: Description" to a cleaner task
                f.write(f"- [ ] {project.strip()}\n \n")

            f.write("\n## 💻 Coding Lab\n")
            # Assuming projects are separated by ';' in your database
            for project in row['non_coding_projects'].split(';'):
                    # Reformat "Name: Description" to a cleaner task
                f.write(f"- [ ] {project.strip()}\n \n")
            
            f.write("\n## 💻 Coding Lab\n")
            # Assuming projects are separated by ';' in your database
            for project in row['online_problem_sets'].split(';'):
                    # Reformat "Name: Description" to a cleaner task
                f.write(f"- [ ] {project.strip()}\n \n")
            
            # 4. Resources (Bullet points with Emojis)
            f.write("\n## 🔗 Best video Resources\n")
            for resource in row['video_resources'].split(';'):
                if resource.strip():
                    f.write(f"- 📖 {resource.strip()}\n \n")
            
            f.write("\n## 🔗 Best text Resources\n")
            for resource in row['text_resources'].split(';'):
                if resource.strip():
                    f.write(f"- 📖 {resource.strip()}\n \n")
                    
            f.write("\n---\n\n")
        

if __name__ == "__main__":
    topic = "Surds and exponents"

    prompt = f"""Generate a study database for:
    {topic}. If there are multiple topics make a row for each topic.
    Make sure that at least five different items are generated for each category and please be detailed. 
    Separate each item with a ;
    For all resources (Text and video) pleast attach a links
    """

    class Module(BaseModel):
        topic:str
        prerequisites: str
        sub_topics: str
        coding_projects: str
        non_coding_projects: str
        online_problem_sets:str
        video_resources: str
        text_resources: str

    client = genai.Client(api_key=os.environ.get("GENAI_API_KEY") )

    response = client.models.generate_content(model = "gemini-2.5-flash",
                                    contents = prompt,
                                    config = {'response_mime_type':'application/json',
                                                'response_schema':list[Module]})

    # print(response)
    
    data = [m.model_dump() for m in response.parsed] #Converts response to a list of dictionaries.
    df_out = pl.DataFrame(data)
    if data_path.exists():
        df_overall = pl.read_parquet(data_path)
        df_overall = df_overall.vstack(df_out) #Need to assign since polars is not inplace
    else:
        df_overall = df_out
    
    df_overall.write_parquet("data.parquet")
        
    df_overall.write_excel("Output.xlsx")

    print("Success! File created using polars.")


    #Unhardcode this
    
    convert_to_markdown(df_overall)
    

    print("STUDY_GUIDE.md created! Open it in VS Code or Obsidian.")


 
#Need to adjust script so it adds to a main database rather than creating a new database eacht ime!



