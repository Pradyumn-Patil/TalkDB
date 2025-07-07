from google import genai

GEMINI_API_KEY = "AIzaSyD0eekHPdt7IjkIBC569y66caNqmLglyr0"
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)